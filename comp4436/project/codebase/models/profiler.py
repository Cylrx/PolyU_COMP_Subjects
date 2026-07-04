"""Build a dual-device profile cache from real measurements."""

from __future__ import annotations

from collections.abc import Callable
import copy
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.compat import require_torch
from core.console import console, success
from core.types import ModelVariant, ProfileCache, ProfileEntry, ProfileMetadata
from models.compression import CompressionStrategy
from models.runners import InferenceRunner, VariantBundle, build_torch_variant_bundle


def build_profile(
    base_model: Any,
    compressions: list[CompressionStrategy],
    test_dataset: Any,
    input_shape: tuple[int, ...],
    train_dataset: Any | None = None,
    fine_tune_variant: Callable[[Any, Any], Any] | None = None,
    edge_device: str = "cpu",
    cloud_device: str = "cuda",
    description: str = "Custom profile",
    n_samples: int | None = None,
    artifacts_dir: str | Path = "cache/artifacts",
    edge_latency_scale_factor: float = 1.0,
    cloud_latency_scale_factor: float = 1.0,
) -> ProfileCache:
    """Build a complete profile cache from real model inference.

    1. Create compressed variants from base_model
    2. Profile each variant on edge and cloud devices
    3. Count MACs per variant via thop
    4. Return ProfileCache with full metadata

    Args:
        base_model: Trained model (e.g. ResNet-152).
        compressions: List of compression strategies to apply.
        test_dataset: Pre-constructed test dataset with transforms applied.
        input_shape: Model input shape without batch dim, e.g. (3, 32, 32).
        train_dataset: Optional training dataset used for pruning recovery and
            quantization calibration.
        fine_tune_variant: Optional callback that fine-tunes a compressed torch
            model on the provided train_dataset before profiling.
        edge_device: Device used for edge execution measurements.
        cloud_device: Device used for cloud execution measurements.
        description: Human-readable description for metadata.
        n_samples: Limit profiling to first N samples (None = full test set).
    """
    _validate_latency_scale_factor("edge_latency_scale_factor", edge_latency_scale_factor)
    _validate_latency_scale_factor("cloud_latency_scale_factor", cloud_latency_scale_factor)

    require_torch("profiling")
    import torch

    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

    console.print()
    console.rule("[bold]Profiling Model Variants")

    if n_samples is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(n_samples, len(test_dataset))))

    actual_n = len(test_dataset)

    calibration_source = train_dataset if train_dataset is not None else test_dataset
    calibration_dataset = _calibration_subset(calibration_source)
    session_artifacts_dir = Path(artifacts_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build variant list: base + compressed
    variants: list[VariantBundle] = [
        build_torch_variant_bundle("full", base_model, edge_device=edge_device, cloud_device=cloud_device)
    ]
    for strategy in compressions:
        console.print(f"  Compressing: [cyan]{strategy.name}[/]")
        bundle = _build_variant_bundle(
            strategy=strategy,
            base_model=base_model,
            train_dataset=train_dataset,
            fine_tune_variant=fine_tune_variant,
            calibration_dataset=calibration_dataset,
            input_shape=input_shape,
            edge_device=edge_device,
            cloud_device=cloud_device,
            artifacts_dir=session_artifacts_dir / strategy.name,
        )
        variants.append(bundle)

    # Profile each variant
    entries: dict[tuple[int, str], ProfileEntry] = {}
    model_info: dict[str, ModelVariant] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        for bundle in variants:
            task = progress.add_task(f"Profiling {bundle.name}", total=actual_n)
            try:
                edge_records, accuracy, edge_avg_latency = _profile_single_variant(
                    runner=bundle.edge_runner,
                    dataset=test_dataset,
                    progress=progress,
                    task=task,
                    collect_predictions=True,
                    latency_scale_factor=edge_latency_scale_factor,
                )
                cloud_records, _, cloud_avg_latency = _profile_single_variant(
                    runner=bundle.cloud_runner,
                    dataset=test_dataset,
                    progress=progress,
                    task=None,
                    collect_predictions=False,
                    latency_scale_factor=cloud_latency_scale_factor,
                )

                for idx, edge_entry in edge_records.items():
                    cloud_entry = cloud_records[idx]
                    entries[(idx, bundle.name)] = ProfileEntry(
                        prediction=edge_entry.prediction,
                        label=edge_entry.label,
                        correct=edge_entry.correct,
                        edge_latency=edge_entry.edge_latency,
                        cloud_latency=cloud_entry.cloud_latency,
                    )

                macs = _count_macs(bundle.macs_model, input_shape, "cpu")

                model_info[bundle.name] = ModelVariant(
                    name=bundle.name,
                    accuracy=accuracy,
                    edge_avg_latency=edge_avg_latency,
                    cloud_avg_latency=cloud_avg_latency,
                    macs=macs,
                )

                success(
                    f"{bundle.name}: acc={accuracy:.2%}, "
                    f"edge={edge_avg_latency * 1000:.2f}ms, "
                    f"cloud={cloud_avg_latency * 1000:.2f}ms, "
                    f"MACs={macs:,}"
                )
            finally:
                bundle.edge_runner.close()
                bundle.cloud_runner.close()

    torch_version = torch.__version__

    metadata = ProfileMetadata(
        kind="profiled",
        n_samples=actual_n,
        model_variants=[bundle.name for bundle in variants],
        edge_device=edge_device,
        cloud_device=cloud_device,
        description=description,
        created_at=datetime.now(timezone.utc).isoformat(),
        torch_version=torch_version,
    )

    profile = ProfileCache(entries=entries, model_info=model_info, metadata=metadata)
    console.print()
    success(f"Profile complete: {actual_n} samples × {len(variants)} variants")
    return profile


def _build_variant_bundle(
    strategy: CompressionStrategy,
    base_model: Any,
    train_dataset: Any | None,
    fine_tune_variant: Callable[[Any, Any], Any] | None,
    calibration_dataset: Any,
    input_shape: tuple[int, ...],
    edge_device: str,
    cloud_device: str,
    artifacts_dir: Path,
) -> VariantBundle:
    compress_model = getattr(strategy, "compress_model", None)
    if callable(compress_model) and train_dataset is not None and fine_tune_variant is not None:
        model = compress_model(base_model)
        console.print(f"    Fine-tuning: [cyan]{strategy.name}[/]")
        model = fine_tune_variant(model, train_dataset)
        return build_torch_variant_bundle(
            strategy.name,
            model,
            edge_device=edge_device,
            cloud_device=cloud_device,
        )

    return strategy.build_variant(
        base_model=base_model,
        calibration_dataset=calibration_dataset,
        input_shape=input_shape,
        edge_device=edge_device,
        cloud_device=cloud_device,
        artifacts_dir=artifacts_dir,
    )


def _profile_single_variant(
    runner: InferenceRunner,
    dataset: Any,
    progress: Any,
    task: Any,
    collect_predictions: bool,
    latency_scale_factor: float,
) -> tuple[dict[int, ProfileEntry], float, float]:
    """Profile one model variant on the dataset.

    Returns (entries_dict, accuracy, avg_latency).
    """
    entries: dict[int, ProfileEntry] = {}
    correct_count = 0
    total_latency = 0.0

    for idx in range(len(dataset)):
        image, label = dataset[idx]

        start = time.perf_counter()
        output = runner.predict(image)
        elapsed = _scale_latency(
            time.perf_counter() - start,
            latency_scale_factor,
        )

        if collect_predictions:
            prediction = output.argmax(dim=1).item()
            is_correct = prediction == label
        else:
            prediction = -1
            is_correct = False

        entries[idx] = ProfileEntry(
            prediction=prediction,
            label=label,
            correct=is_correct,
            edge_latency=elapsed if collect_predictions else 0.0,
            cloud_latency=elapsed if not collect_predictions else 0.0,
        )

        if collect_predictions and is_correct:
            correct_count += 1
        total_latency += elapsed

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    n = len(dataset)
    accuracy = correct_count / n if collect_predictions and n > 0 else 0.0
    avg_latency = total_latency / n if n > 0 else 0.0

    return entries, accuracy, avg_latency


def _count_macs(model: Any, input_shape: tuple[int, ...], device: str = "cpu") -> int:
    """Count multiply-accumulate operations using thop."""
    require_torch("MACs counting")
    import torch

    from thop import profile as thop_profile

    model_copy = copy.deepcopy(model).to(device)
    dummy = torch.randn(1, *input_shape, device=device)
    macs, _ = thop_profile(model_copy, inputs=(dummy,), verbose=False)
    return int(macs)


def _calibration_subset(dataset: Any, max_samples: int = 256) -> Any:
    """Use the first few profile samples as the quantization calibration set."""
    require_torch("profiling")
    import torch

    return torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))


def _validate_latency_scale_factor(name: str, latency_scale_factor: float) -> None:
    if latency_scale_factor < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _scale_latency(latency: float, latency_scale_factor: float) -> float:
    return latency * latency_scale_factor

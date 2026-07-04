from __future__ import annotations

from pathlib import Path
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("thop")
pytest.importorskip("torch_pruning")

from core.types import ProfileEntry
from models.compression import DynamicQuantization, StructuredPruning
from models.profiler import _count_macs, build_profile
from models.resnet import RESNET_ARCHITECTURE, build_resnet18
from models.runners import VariantBundle, build_torch_variant_bundle


class _OneSampleDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        del idx
        return torch.randn(3, 32, 32), 0


class _IdentityStrategy:
    @property
    def name(self) -> str:
        return "copy"

    def build_variant(
        self,
        base_model: object,
        calibration_dataset: object,
        input_shape: tuple[int, ...],
        edge_device: str,
        cloud_device: str,
        artifacts_dir: Path,
    ) -> VariantBundle:
        del calibration_dataset, input_shape, artifacts_dir
        return build_torch_variant_bundle(
            self.name,
            base_model,
            edge_device=edge_device,
            cloud_device=cloud_device,
        )


class _FineTunableIdentityStrategy:
    @property
    def name(self) -> str:
        return "copy"

    def compress_model(self, model: object) -> object:
        return copy.deepcopy(model)

    def build_variant(
        self,
        base_model: object,
        calibration_dataset: object,
        input_shape: tuple[int, ...],
        edge_device: str,
        cloud_device: str,
        artifacts_dir: Path,
    ) -> VariantBundle:
        del base_model, calibration_dataset, input_shape, edge_device, cloud_device, artifacts_dir
        raise AssertionError("build_variant should not be used when fine-tuning a torch-backed variant")


def test_structured_pruning_builds_runner_bundle_for_cuda_model(tmp_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18().to(device)
    dataset = _OneSampleDataset()

    bundle = StructuredPruning(
        amount=0.3,
        example_input_shape=(3, 32, 32),
    ).build_variant(
        base_model=model,
        calibration_dataset=dataset,
        input_shape=(3, 32, 32),
        edge_device="cpu",
        cloud_device=device,
        artifacts_dir=tmp_path,
    )

    assert bundle.name == "structured_30"
    assert bundle.edge_runner.predict(dataset[0][0]).shape == (1, 10)
    assert bundle.cloud_runner.predict(dataset[0][0]).shape == (1, 10)


def test_count_macs_moves_model_to_requested_device() -> None:
    source_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18().to(source_device)

    macs = _count_macs(model, (3, 32, 32), device="cpu")

    assert macs > 0


def test_build_profile_uses_variant_bundles(tmp_path: Path) -> None:
    model = build_resnet18().to("cpu")
    dataset = _OneSampleDataset()

    profile = build_profile(
        base_model=model,
        compressions=[_IdentityStrategy()],
        test_dataset=dataset,
        input_shape=(3, 32, 32),
        edge_device="cpu",
        cloud_device="cpu",
        description="test",
        n_samples=1,
        artifacts_dir=tmp_path,
    )

    assert profile.model_names == ["copy", "full"]
    assert profile.metadata.model_variants == ["full", "copy"]


def test_build_profile_applies_separate_latency_scale_factors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = build_resnet18().to("cpu")
    dataset = _OneSampleDataset()

    call_count = {"value": 0}

    def _fake_profile_single_variant(
        runner: object,
        dataset: object,
        progress: object,
        task: object,
        collect_predictions: bool,
        latency_scale_factor: float,
    ) -> tuple[dict[int, ProfileEntry], float, float]:
        del runner, dataset, progress, task
        call_count["value"] += 1
        base_latency = 0.01 if collect_predictions else 0.02
        scaled_latency = base_latency * latency_scale_factor
        return (
            {
                0: ProfileEntry(
                    prediction=0 if collect_predictions else -1,
                    label=0,
                    correct=collect_predictions,
                    edge_latency=scaled_latency if collect_predictions else 0.0,
                    cloud_latency=scaled_latency if not collect_predictions else 0.0,
                )
            },
            1.0 if collect_predictions else 0.0,
            scaled_latency,
        )

    monkeypatch.setattr("models.profiler._profile_single_variant", _fake_profile_single_variant)
    monkeypatch.setattr("models.profiler._count_macs", lambda *args, **kwargs: 123)

    profile = build_profile(
        base_model=model,
        compressions=[_IdentityStrategy()],
        test_dataset=dataset,
        input_shape=(3, 32, 32),
        edge_device="cpu",
        cloud_device="cpu",
        description="test",
        n_samples=1,
        artifacts_dir=tmp_path,
        edge_latency_scale_factor=3.0,
        cloud_latency_scale_factor=4.0,
    )

    assert call_count["value"] == 4
    for model_name in profile.model_names:
        entry = profile.lookup(0, model_name)
        assert entry.edge_latency == pytest.approx(0.03)
        assert entry.cloud_latency == pytest.approx(0.08)

        model_info = next(m for m in profile.available_models if m.name == model_name)
        assert model_info.edge_avg_latency == pytest.approx(0.03)
        assert model_info.cloud_avg_latency == pytest.approx(0.08)


def test_quantized_int8_fails_fast_without_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise() -> object:
        raise ImportError("missing quant backends")

    monkeypatch.setattr("models.compression._require_quantization_backends", _raise)

    model = build_resnet18().to("cpu")
    dataset = _OneSampleDataset()

    with pytest.raises(ImportError, match="missing quant backends"):
        DynamicQuantization().build_variant(
            base_model=model,
            calibration_dataset=dataset,
            input_shape=(3, 32, 32),
            edge_device="cpu",
            cloud_device="cuda",
            artifacts_dir=tmp_path,
        )


def test_default_resnet_architecture_is_resnet152() -> None:
    assert RESNET_ARCHITECTURE == "resnet152"


def test_build_profile_fine_tunes_torch_variants_before_profiling(tmp_path: Path) -> None:
    model = build_resnet18().to("cpu")
    dataset = _OneSampleDataset()
    calls: list[int] = []

    def _fine_tune(variant: object, train_dataset: object) -> object:
        calls.append(len(train_dataset))  # type: ignore[arg-type]
        return variant

    profile = build_profile(
        base_model=model,
        compressions=[_FineTunableIdentityStrategy()],
        train_dataset=dataset,
        test_dataset=dataset,
        input_shape=(3, 32, 32),
        fine_tune_variant=_fine_tune,
        edge_device="cpu",
        cloud_device="cpu",
        description="test",
        n_samples=1,
        artifacts_dir=tmp_path,
    )

    assert profile.model_names == ["copy", "full"]
    assert calls == [1]

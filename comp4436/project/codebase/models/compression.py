"""Model compression strategies and per-device profiling artifacts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Protocol

from core.compat import require_torch
from models.runners import VariantBundle, build_onnxruntime_runner, build_torch_variant_bundle


class CompressionStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def build_variant(
        self,
        base_model: Any,
        calibration_dataset: Any,
        input_shape: tuple[int, ...],
        edge_device: str,
        cloud_device: str,
        artifacts_dir: Path,
    ) -> VariantBundle: ...


class TorchCompressionStrategy(CompressionStrategy, Protocol):
    def compress_model(self, model: Any) -> Any: ...


class DynamicQuantization:
    """INT8 quantization via ONNX Runtime on CPU and TensorRT EP on CUDA."""

    @property
    def name(self) -> str:
        return "quantized_int8"

    def build_variant(
        self,
        base_model: Any,
        calibration_dataset: Any,
        input_shape: tuple[int, ...],
        edge_device: str,
        cloud_device: str,
        artifacts_dir: Path,
    ) -> VariantBundle:
        require_torch("quantized profiling")
        import torch

        ortq = _require_quantization_backends()
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_cpu = copy.deepcopy(base_model).to("cpu").eval()
        float_onnx_path = artifacts_dir / "model_fp32.onnx"
        quantized_onnx_path = artifacts_dir / "model_int8.onnx"

        dummy = torch.randn(1, *input_shape)
        torch.onnx.export(
            model_cpu,
            dummy,
            str(float_onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )
        calibration_reader = _CalibrationDataReader(
            dataset=calibration_dataset,
            input_name="input",
        )
        ortq.quantize_static(
            str(float_onnx_path),
            str(quantized_onnx_path),
            calibration_reader,
            quant_format=ortq.QuantFormat.QDQ,
            activation_type=ortq.QuantType.QInt8,
            weight_type=ortq.QuantType.QInt8,
        )

        return VariantBundle(
            name=self.name,
            edge_runner=build_onnxruntime_runner(
                quantized_onnx_path,
                edge_device,
                cache_dir=artifacts_dir / "edge_cache",
            ),
            cloud_runner=build_onnxruntime_runner(
                quantized_onnx_path,
                cloud_device,
                cache_dir=artifacts_dir / "cloud_cache",
            ),
            macs_model=model_cpu,
        )


class UnstructuredPruning:
    """L1 magnitude-based unstructured pruning."""

    def __init__(self, amount: float) -> None:
        if not 0 < amount < 1:
            raise ValueError(f"Pruning amount must be in (0, 1), got {amount}")
        self.amount = amount

    @property
    def name(self) -> str:
        return f"pruned_{int(self.amount * 100)}"

    def compress_model(self, model: Any) -> Any:
        require_torch("unstructured pruning")
        import torch.nn as nn
        import torch.nn.utils.prune as prune

        model_copy = copy.deepcopy(model)
        for module in model_copy.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=self.amount)
                prune.remove(module, "weight")
        return model_copy

    def build_variant(
        self,
        base_model: Any,
        calibration_dataset: Any,
        input_shape: tuple[int, ...],
        edge_device: str,
        cloud_device: str,
        artifacts_dir: Path,
    ) -> VariantBundle:
        del calibration_dataset, input_shape, artifacts_dir
        return build_torch_variant_bundle(
            self.name,
            self.compress_model(base_model),
            edge_device=edge_device,
            cloud_device=cloud_device,
        )


class StructuredPruning:
    """Channel-level structured pruning via torch-pruning.

    Actually removes channels from conv layers, reducing model dimensions
    and real inference MACs.
    """

    def __init__(self, amount: float, example_input_shape: tuple[int, ...]) -> None:
        if not 0 < amount < 1:
            raise ValueError(f"Pruning amount must be in (0, 1), got {amount}")
        self.amount = amount
        self.example_input_shape = example_input_shape

    @property
    def name(self) -> str:
        return f"structured_{int(self.amount * 100)}"

    def compress_model(self, model: Any) -> Any:
        require_torch("structured pruning")
        import torch
        import torch_pruning as tp

        model_copy = copy.deepcopy(model)
        model_copy.eval()

        parameter = next(model_copy.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        example_input = torch.randn(1, *self.example_input_shape, device=device)
        importance = tp.importance.MagnitudeImportance()

        ignored_layers = []
        for module in model_copy.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 10:
                ignored_layers.append(module)

        pruner = tp.pruner.MagnitudePruner(
            model_copy,
            example_input,
            importance,
            pruning_ratio=self.amount,
            ignored_layers=ignored_layers,
        )
        pruner.step()
        return model_copy

    def build_variant(
        self,
        base_model: Any,
        calibration_dataset: Any,
        input_shape: tuple[int, ...],
        edge_device: str,
        cloud_device: str,
        artifacts_dir: Path,
    ) -> VariantBundle:
        del calibration_dataset, input_shape, artifacts_dir
        return build_torch_variant_bundle(
            self.name,
            self.compress_model(base_model),
            edge_device=edge_device,
            cloud_device=cloud_device,
        )


class CompositeCompression:
    """Chain multiple compression strategies sequentially."""

    def __init__(self, strategies: list[CompressionStrategy], name: str) -> None:
        self._strategies = strategies
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def build_variant(
        self,
        base_model: Any,
        calibration_dataset: Any,
        input_shape: tuple[int, ...],
        edge_device: str,
        cloud_device: str,
        artifacts_dir: Path,
    ) -> VariantBundle:
        del calibration_dataset, input_shape, artifacts_dir
        result = base_model
        for strategy in self._strategies:
            compress_model = getattr(strategy, "compress_model", None)
            if compress_model is None:
                raise TypeError(
                    "CompositeCompression only supports strategies backed by torch models."
                )
            result = compress_model(result)
        return build_torch_variant_bundle(
            self.name,
            result,
            edge_device=edge_device,
            cloud_device=cloud_device,
        )


class _CalibrationDataReader:
    """ONNX Runtime calibration reader over a small dataset subset."""

    def __init__(self, dataset: Any, input_name: str) -> None:
        if len(dataset) == 0:
            raise ValueError("Calibration dataset is empty.")
        self._input_name = input_name
        self._items = []
        for idx in range(len(dataset)):
            image, _ = dataset[idx]
            batch = image.unsqueeze(0).detach().cpu().numpy()
            self._items.append({self._input_name: batch})
        self._index = 0

    def get_next(self) -> dict[str, Any] | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        return item

    def rewind(self) -> None:
        self._index = 0


def _require_quantization_backends() -> Any:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "quantized_int8 requires the optional ML backends: install 'onnx' and "
            "'onnxruntime-gpu', and ensure TensorRT Execution Provider is available."
        ) from exc

    try:
        from onnxruntime import quantization as ortq
    except ImportError as exc:
        raise ImportError(
            "quantized_int8 requires the optional ML backends: install 'onnx' and "
            "'onnxruntime-gpu', and ensure TensorRT Execution Provider is available."
        ) from exc

    return ortq


def default_compressions(input_shape: tuple[int, ...]) -> list[CompressionStrategy]:
    """Build the standard set of compression strategies for a given input shape."""
    return [
        DynamicQuantization(),
        UnstructuredPruning(amount=0.3),
        UnstructuredPruning(amount=0.6),
        StructuredPruning(amount=0.3, example_input_shape=input_shape),
        StructuredPruning(amount=0.5, example_input_shape=input_shape),
    ]

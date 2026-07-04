"""Execution runners for per-device profiling backends."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.compat import require_torch


class InferenceRunner(Protocol):
    def predict(self, image: Any) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class VariantBundle:
    name: str
    edge_runner: InferenceRunner
    cloud_runner: InferenceRunner
    macs_model: Any


class TorchRunner:
    """Run a torch.nn.Module on a fixed device and return CPU logits."""

    def __init__(self, model: Any, device: str) -> None:
        require_torch("torch profiling runner")
        self._model = model.to(device)
        self._device = device
        self._model.eval()

    def predict(self, image: Any) -> Any:
        require_torch("torch profiling runner")
        import torch

        with torch.no_grad():
            batch = image.unsqueeze(0).to(self._device)
            return self._model(batch).detach().cpu()

    def close(self) -> None:
        return None


class OnnxRuntimeRunner:
    """Run an ONNX model through ONNX Runtime and return CPU logits."""

    def __init__(self, model_path: Path, providers: list[Any]) -> None:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, image: Any) -> Any:
        require_torch("onnxruntime profiling runner")
        import torch

        batch = image.unsqueeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = self._session.run(None, {self._input_name: batch})
        return torch.from_numpy(outputs[0])

    def close(self) -> None:
        return None


def build_torch_variant_bundle(
    name: str,
    model: Any,
    edge_device: str,
    cloud_device: str,
) -> VariantBundle:
    """Create per-device torch runners from a base model."""
    return VariantBundle(
        name=name,
        edge_runner=TorchRunner(copy.deepcopy(model), edge_device),
        cloud_runner=TorchRunner(copy.deepcopy(model), cloud_device),
        macs_model=copy.deepcopy(model).to("cpu").eval(),
    )


def build_onnxruntime_runner(
    model_path: Path,
    device: str,
    cache_dir: Path,
) -> InferenceRunner:
    """Build an ONNX Runtime runner for the requested device.

    CPU uses the CPU execution provider.
    CUDA uses TensorRT EP with CUDA fallback disabled at construction time if the
    TensorRT provider is unavailable.
    """
    import onnxruntime as ort

    if device.startswith("cuda"):
        available = set(ort.get_available_providers())
        if "TensorrtExecutionProvider" not in available:
            raise ImportError(
                "TensorRT Execution Provider is unavailable. "
                "Install onnxruntime-gpu with TensorRT support to profile quantized_int8 on CUDA."
            )
        if "CUDAExecutionProvider" not in available:
            raise ImportError(
                "CUDAExecutionProvider is unavailable. "
                "Install an ONNX Runtime GPU build to profile quantized_int8 on CUDA."
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers: list[Any] = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_int8_enable": True,
                },
            ),
            "CUDAExecutionProvider",
        ]
        return OnnxRuntimeRunner(model_path, providers=providers)

    return OnnxRuntimeRunner(model_path, providers=["CPUExecutionProvider"])

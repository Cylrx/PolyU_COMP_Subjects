"""Torch availability checks and environment validation.

All torch-dependent code should call require_torch() before importing torch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

_torch_available: bool | None = None


def is_torch_available() -> bool:
    """Check if PyTorch is importable. Result is cached after first call."""
    global _torch_available
    if _torch_available is None:
        try:
            import torch  # noqa: F401

            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def require_torch(feature: str = "this feature") -> None:
    """Raise ImportError with install instructions if torch is not available."""
    if not is_torch_available():
        raise ImportError(
            f"PyTorch is required for {feature}. "
            "Install with: uv add --optional ml torch torchvision"
        )


@dataclass(frozen=True)
class EnvironmentStatus:
    python_version: str
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    torchvision_available: bool
    device: str

    @property
    def can_profile(self) -> bool:
        return self.torch_available and self.torchvision_available


def check_environment() -> EnvironmentStatus:
    """Run all environment checks. Never raises — returns status."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    torch_available = is_torch_available()
    torch_version: str | None = None
    cuda_available = False
    cuda_version: str | None = None

    if torch_available:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda

    # Check torchvision accessibility
    torchvision_available = False
    if torch_available:
        try:
            import torchvision  # noqa: F401

            torchvision_available = True
        except ImportError:
            pass

    device = "cuda" if cuda_available else "cpu"

    return EnvironmentStatus(
        python_version=python_version,
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        torchvision_available=torchvision_available,
        device=device,
    )

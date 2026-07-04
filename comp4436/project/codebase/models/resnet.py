"""ResNet model definition and training for CIFAR-10.

Uses torchvision's ResNet family adapted for 32x32 CIFAR-10 inputs:
  - 3x3 conv1 (stride=1, padding=1) instead of 7x7 (stride=2)
  - No early max-pooling

All torch imports are guarded — this module is only usable when the [ml]
optional dependency group is installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.compat import require_torch
from core.console import console, success

CIFAR10_INPUT_SHAPE = (3, 32, 32)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
RESNET_ARCHITECTURE = "resnet152"
RESNET_DISPLAY_NAME = "ResNet-152"


def build_resnet(architecture: str = RESNET_ARCHITECTURE) -> Any:
    """Build a torchvision ResNet adapted for CIFAR-10 (3-channel, 32x32)."""
    require_torch(RESNET_DISPLAY_NAME)
    import torch.nn as nn
    import torchvision.models as models

    try:
        builder = getattr(models, architecture)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torchvision ResNet architecture: {architecture}") from exc

    model = builder(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_resnet152() -> Any:
    """Build the default CIFAR ResNet backbone."""
    return build_resnet(RESNET_ARCHITECTURE)


def build_resnet18() -> Any:
    """Backward-compatible helper for the smaller ResNet-18 variant."""
    return build_resnet("resnet18")


def build_cifar10_datasets(data_dir: str = "./data") -> tuple[Any, Any]:
    """Return train/test CIFAR-10 datasets with the transforms used in training."""
    require_torch("model training")
    from torchvision import datasets, transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return (
        datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train),
        datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test),
    )


def train_resnet(
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 128,
    device: str = "cpu",
    data_dir: str = "./data",
) -> tuple[Any, float]:
    """Train the default ResNet backbone on CIFAR-10. Returns (model, test_accuracy)."""
    require_torch("model training")
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    console.print()
    console.rule(f"[bold]Training {RESNET_DISPLAY_NAME} on CIFAR-10")

    train_dataset, test_dataset = build_cifar10_datasets(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    model = build_resnet()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("Training", total=epochs)
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            progress.update(epoch_task, advance=1, description=f"Epoch {epoch + 1}/{epochs}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    success(f"Training complete — test accuracy: {accuracy:.2%}")
    return model, accuracy


def fine_tune_resnet(
    model: Any,
    train_dataset: Any,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
) -> Any:
    """Fine-tune a compressed CIFAR ResNet variant on the training split."""
    require_torch("model fine-tuning")
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("Fine-tuning", total=epochs)
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            progress.update(epoch_task, advance=1, description=f"Epoch {epoch + 1}/{epochs}")

    model.eval()
    return model


def save_weights(model: Any, path: Path, metadata: dict[str, Any]) -> None:
    require_torch("saving weights")
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    success(f"Weights saved to {path}")


def load_weights(path: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    require_torch("loading weights")
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint.get("metadata", {})
    architecture = meta.get("architecture", RESNET_ARCHITECTURE)
    model = build_resnet(architecture)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, meta

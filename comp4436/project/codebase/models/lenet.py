"""LeNet-5 model definition and training for MNIST.

Supports a `width` multiplier to scale the model larger for meaningful
latency differences between compressed variants.

All torch imports are guarded — this module is only usable when the [ml]
optional dependency group is installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.compat import require_torch
from core.console import console, success

DEFAULT_WIDTH = 12


def _get_torch():  # type: ignore[no-untyped-def]
    require_torch("model training")
    import torch
    import torch.nn as nn

    return torch, nn


class LeNet5:
    """LeNet-5 with configurable width multiplier.

    width=1:  original LeNet-5  (~400K MACs)
    width=12: wide LeNet        (~66M MACs, meaningful compression effects)
    """

    @staticmethod
    def build(width: int = DEFAULT_WIDTH):  # type: ignore[no-untyped-def]
        torch, nn = _get_torch()

        class _LeNet5(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 6 * width, 5, padding=2)
                self.conv2 = nn.Conv2d(6 * width, 16 * width, 5)
                self.fc1 = nn.Linear(16 * width * 5 * 5, 120 * width)
                self.fc2 = nn.Linear(120 * width, 84 * width)
                self.fc3 = nn.Linear(84 * width, 10)
                self.pool = nn.MaxPool2d(2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):  # type: ignore[no-untyped-def]
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                return x

        return _LeNet5()


def build_resnet18():  # type: ignore[no-untyped-def]
    """ResNet-18 adapted for MNIST (1-channel, 28x28). Backup model."""
    require_torch("ResNet-18")
    import torch.nn as nn
    import torchvision.models as models

    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train_lenet(
    width: int = DEFAULT_WIDTH,
    epochs: int = 15,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
    data_dir: str = "./data",
) -> tuple[Any, float]:
    """Train LeNet-5 on MNIST. Returns (model, test_accuracy)."""
    require_torch("model training")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    console.print()
    console.rule(f"[bold]Training LeNet-5 (width={width}) on MNIST")

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LeNet5.build(width=width)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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


def build_mnist_datasets(data_dir: str = "./data") -> tuple[Any, Any]:
    """Return train/test MNIST datasets with the transforms used in training."""
    require_torch("model training")
    from torchvision import datasets, transforms

    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return (
        datasets.MNIST(data_dir, train=True, download=True, transform=transform_train),
        datasets.MNIST(data_dir, train=False, download=True, transform=transform_test),
    )


def fine_tune_lenet(
    model: Any,
    train_dataset: Any,
    epochs: int = 3,
    lr: float = 0.0005,
    batch_size: int = 64,
    device: str = "cpu",
) -> Any:
    """Fine-tune a compressed LeNet variant on the training split."""
    require_torch("model fine-tuning")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    width = meta.get("width", DEFAULT_WIDTH)
    model = LeNet5.build(width=width)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, meta


def get_weight_metadata(path: Path) -> dict[str, Any]:
    require_torch("reading weight metadata")
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("metadata", {})

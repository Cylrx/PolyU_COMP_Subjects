from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path


class Location(Enum):
    EDGE = auto()
    CLOUD = auto()


# -- Mock profile presets -----------------------------------------------------
# Each entry: (accuracy, edge_latency_s, cloud_latency_s, macs)
#
# Both presets share the same heterogeneous device ordering pattern:
#   Edge  (fast→slow): quant < struct50 < struct30 < prune60 < prune30 < full
#   Cloud (fast→slow): struct50 < struct30 < prune60 < prune30 < full < quant

MockProfileSpec = dict[str, tuple[float, float, float, int]]

RESNET_CIFAR10_PROFILES: MockProfileSpec = {
    "full":           (0.957, 0.0450, 0.0050, 556_000_000),
    "quantized_int8": (0.930, 0.0220, 0.0065, 556_000_000),
    "pruned_30":      (0.925, 0.0430, 0.0048, 556_000_000),
    "pruned_60":      (0.890, 0.0400, 0.0046, 556_000_000),
    "structured_30":  (0.915, 0.0320, 0.0035, 272_000_000),
    "structured_50":  (0.870, 0.0240, 0.0028, 139_000_000),
}

LENET_MNIST_PROFILES: MockProfileSpec = {
    "full":           (0.985, 0.0280, 0.0075, 66_000_000),
    "quantized_int8": (0.975, 0.0095, 0.0095, 66_000_000),
    "pruned_30":      (0.980, 0.0265, 0.0072, 66_000_000),
    "pruned_60":      (0.960, 0.0250, 0.0070, 66_000_000),
    "structured_30":  (0.970, 0.0200, 0.0052, 46_000_000),
    "structured_50":  (0.945, 0.0150, 0.0038, 33_000_000),
}


@dataclass(frozen=True)
class ModelVariant:
    name: str
    accuracy: float
    edge_avg_latency: float
    cloud_avg_latency: float
    macs: int


@dataclass(frozen=True)
class DataSample:
    id: int
    dataset_idx: int
    label: int
    arrival_time: float
    deadline: float


@dataclass(frozen=True)
class ProfileEntry:
    prediction: int
    label: int
    correct: bool
    edge_latency: float
    cloud_latency: float


@dataclass(frozen=True)
class InferenceRequest:
    sample: DataSample
    model_name: str
    location: Location
    start_time: float


@dataclass(frozen=True)
class ProfileMetadata:
    kind: str
    n_samples: int
    model_variants: list[str]
    edge_device: str
    cloud_device: str
    description: str
    created_at: str
    torch_version: str | None


class ProfileCache:
    """Maps (dataset_idx, model_name) -> ProfileEntry.

    Includes metadata and per-model aggregate info (ModelVariant).
    Serializes to portable JSON — no torch needed to load.
    """

    def __init__(
        self,
        entries: dict[tuple[int, str], ProfileEntry],
        model_info: dict[str, ModelVariant],
        metadata: ProfileMetadata,
    ) -> None:
        self._entries = entries
        self._model_info = model_info
        self.metadata = metadata

    @property
    def size(self) -> int:
        return len({idx for idx, _ in self._entries})

    @property
    def model_names(self) -> list[str]:
        return sorted(self._model_info.keys())

    @property
    def available_models(self) -> list[ModelVariant]:
        return list(self._model_info.values())

    @property
    def is_mock(self) -> bool:
        return self.metadata.kind == "mock"

    def lookup(self, dataset_idx: int, model_name: str) -> ProfileEntry:
        return self._entries[(dataset_idx, model_name)]

    def blacklisted(self, disabled_variants: tuple[str, ...] | list[str]) -> ProfileCache:
        disabled = set(disabled_variants)
        if not disabled:
            return self

        kept_names = [name for name in self.metadata.model_variants if name not in disabled]
        if not kept_names:
            raise ValueError("Model variant blacklist removed every variant from the profile")

        model_info = {
            name: variant
            for name, variant in self._model_info.items()
            if name in kept_names
        }
        if not model_info:
            raise ValueError("No model variants remain after applying the blacklist")

        entries = {
            (idx, model_name): entry
            for (idx, model_name), entry in self._entries.items()
            if model_name in model_info
        }
        metadata = ProfileMetadata(
            kind=self.metadata.kind,
            n_samples=self.metadata.n_samples,
            model_variants=kept_names,
            edge_device=self.metadata.edge_device,
            cloud_device=self.metadata.cloud_device,
            description=self.metadata.description,
            created_at=self.metadata.created_at,
            torch_version=self.metadata.torch_version,
        )
        return ProfileCache(entries=entries, model_info=model_info, metadata=metadata)

    # -- JSON serialization ---------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        serialized_entries = {
            f"{idx},{name}": asdict(entry)
            for (idx, name), entry in self._entries.items()
        }
        serialized_model_info = {
            name: asdict(variant)
            for name, variant in self._model_info.items()
        }

        doc = {
            "metadata": asdict(self.metadata),
            "model_info": serialized_model_info,
            "entries": serialized_entries,
        }

        with open(path, "w") as f:
            json.dump(doc, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ProfileCache:
        with open(path) as f:
            doc = json.load(f)

        try:
            metadata = ProfileMetadata(**doc["metadata"])
            model_info = {
                name: ModelVariant(**data)
                for name, data in doc["model_info"].items()
            }
        except TypeError as exc:
            raise ValueError(
                "Unsupported profile schema. Re-profile to generate a dual-device cache."
            ) from exc

        entries: dict[tuple[int, str], ProfileEntry] = {}
        for key, data in doc["entries"].items():
            idx_str, model_name = key.split(",", 1)
            try:
                entries[(int(idx_str), model_name)] = ProfileEntry(**data)
            except TypeError as exc:
                raise ValueError(
                    "Unsupported profile schema. Re-profile to generate a dual-device cache."
                ) from exc

        return cls(entries=entries, model_info=model_info, metadata=metadata)

    # -- Mock for testing (no torch needed) -----------------------------------

    @classmethod
    def mock(
        cls,
        n_samples: int = 100,
        model_names: list[str] | None = None,
        model_profiles: MockProfileSpec | None = None,
        seed: int = 0,
        edge_latency_scale_factor: float = 1.0,
        cloud_latency_scale_factor: float = 1.0,
    ) -> ProfileCache:
        """Create a deterministic mock dual-device cache.

        The predefined model templates intentionally expose:
        - meaningful accuracy spread across variants
        - meaningful latency spread across variants
        - heterogeneous CPU/GPU ordering across variants

        Sample-level correctness is generated by Bernoulli draws from each
        variant's predefined average accuracy. Sample-level latencies are
        generated from each device's predefined mean latency plus small jitter.
        The stored model-level stats are then recomputed from the sampled
        sample-level values so the realized cache remains self-consistent.

        Args:
            model_profiles: Override the default profile templates. Use
                RESNET_CIFAR10_PROFILES or LENET_MNIST_PROFILES, or provide
                custom values. Defaults to RESNET_CIFAR10_PROFILES.
        """
        if model_profiles is None:
            model_profiles = RESNET_CIFAR10_PROFILES
        if model_names is None:
            model_names = list(model_profiles.keys())
        if edge_latency_scale_factor < 0.0:
            raise ValueError("edge_latency_scale_factor must be non-negative")
        if cloud_latency_scale_factor < 0.0:
            raise ValueError("cloud_latency_scale_factor must be non-negative")

        rng = random.Random(seed)

        default_profile = (0.900, 0.0350, 0.0045, 400_000_000)

        entries: dict[tuple[int, str], ProfileEntry] = {}
        correct_counts: dict[str, int] = {m: 0 for m in model_names}
        edge_latency_sums: dict[str, float] = {m: 0.0 for m in model_names}
        cloud_latency_sums: dict[str, float] = {m: 0.0 for m in model_names}

        for idx in range(n_samples):
            label = idx % 10
            for model_name in model_names:
                acc, edge_base_lat, cloud_base_lat, _ = model_profiles.get(
                    model_name, default_profile
                )
                correct = rng.random() < acc
                prediction = label if correct else (label + rng.randint(1, 9)) % 10
                edge_latency = (
                    edge_base_lat * rng.uniform(0.97, 1.03) * edge_latency_scale_factor
                )
                cloud_latency = (
                    cloud_base_lat * rng.uniform(0.97, 1.03) * cloud_latency_scale_factor
                )
                entries[(idx, model_name)] = ProfileEntry(
                    prediction=prediction,
                    label=label,
                    correct=correct,
                    edge_latency=edge_latency,
                    cloud_latency=cloud_latency,
                )
                if correct:
                    correct_counts[model_name] += 1
                edge_latency_sums[model_name] += edge_latency
                cloud_latency_sums[model_name] += cloud_latency

        model_info: dict[str, ModelVariant] = {}
        for model_name in model_names:
            acc, _, _, macs = model_profiles.get(model_name, default_profile)
            model_info[model_name] = ModelVariant(
                name=model_name,
                accuracy=correct_counts[model_name] / n_samples,
                edge_avg_latency=edge_latency_sums[model_name] / n_samples,
                cloud_avg_latency=cloud_latency_sums[model_name] / n_samples,
                macs=macs,
            )

        metadata = ProfileMetadata(
            kind="mock",
            n_samples=n_samples,
            model_variants=model_names,
            edge_device="mock-cpu",
            cloud_device="mock-cuda",
            description="Mock dual-device profile with faithful simulated spread",
            created_at=datetime.now(timezone.utc).isoformat(),
            torch_version=None,
        )

        return cls(entries=entries, model_info=model_info, metadata=metadata)

from __future__ import annotations

from config import SimulationConfig
from core.types import DataSample, ModelVariant


def make_sample(
    id: int = 0,
    arrival_time: float = 0.0,
    deadline: float = 10.0,
    dataset_idx: int | None = None,
    label: int = 0,
) -> DataSample:
    return DataSample(
        id=id,
        dataset_idx=dataset_idx if dataset_idx is not None else id,
        label=label,
        arrival_time=arrival_time,
        deadline=deadline,
    )


def make_config(**overrides: object) -> SimulationConfig:
    defaults = dict(
        duration=10.0,
        deadline_budget=0.5,
        queue_capacity=50,
        edge_model_load_time=0.05,
        cloud_capacity=1,
        network_base_rtt=0.1,
        network_jitter_std=0.0,
        seed=42,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


MOCK_MODELS = [
    ModelVariant(
        name="full",
        accuracy=0.935,
        edge_avg_latency=0.045,
        cloud_avg_latency=0.005,
        macs=556_000_000,
    ),
    ModelVariant(
        name="quantized_int8",
        accuracy=0.930,
        edge_avg_latency=0.022,
        cloud_avg_latency=0.0065,
        macs=556_000_000,
    ),
]

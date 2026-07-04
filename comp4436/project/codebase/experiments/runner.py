from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import SimulationConfig
from core.cloud import CloudNode
from core.console import SampleProgressDisplay
from core.edge import EdgeNode
from core.network import NetworkModel
from core.pipeline import Pipeline
from core.queue import ProcessingQueue
from core.types import ModelVariant, ProfileCache
from evaluation.metrics import MetricsCollector, MetricsSummary

if TYPE_CHECKING:
    from strategies.protocols import (
        AdmissionStrategy,
        DispatchStrategy,
    )


@dataclass
class ExperimentConfig:
    name: str
    description: str
    simulation: SimulationConfig
    admission: AdmissionStrategy
    dispatcher: DispatchStrategy
    arrival_times: tuple[float, ...]


class ExperimentRunner:
    """Runs one or more experiments against a shared ProfileCache."""

    def __init__(
        self,
        profile: ProfileCache,
        available_models: list[ModelVariant],
        metric_interval: float = 5.0,
    ) -> None:
        self._profile = profile
        self._available_models = available_models
        self._metric_interval = metric_interval

    def run_one(self, config: ExperimentConfig) -> MetricsSummary:
        sim = config.simulation
        net = NetworkModel(
            base_rtt=sim.network_base_rtt,
            jitter_std=sim.network_jitter_std,
            seed=sim.seed,
        )
        pipeline = Pipeline(
            profile=self._profile,
            queue=ProcessingQueue(sim.queue_capacity),
            edge=EdgeNode(model_load_time=sim.edge_model_load_time),
            cloud=CloudNode(capacity=sim.cloud_capacity, network=net),
            admission=config.admission,
            dispatcher=config.dispatcher,
            arrival_times=config.arrival_times,
            available_models=self._available_models,
            metrics=MetricsCollector(),
            config=sim,
            metric_interval=self._metric_interval,
        )
        dispatcher = config.dispatcher
        progress_label = f"  {config.name}"
        with SampleProgressDisplay(
            label=progress_label,
            total_samples=len(config.arrival_times),
        ) as progress:
            if hasattr(dispatcher, "set_progress_reporter"):
                dispatcher.set_progress_reporter(progress.update)
            try:
                return pipeline.run()
            finally:
                if hasattr(dispatcher, "set_progress_reporter"):
                    dispatcher.set_progress_reporter(None)

    def run_all(
        self, configs: list[ExperimentConfig]
    ) -> dict[str, MetricsSummary]:
        return {cfg.name: self.run_one(cfg) for cfg in configs}

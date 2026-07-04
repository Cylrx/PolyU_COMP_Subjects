from __future__ import annotations

from config import SimulationConfig
from core.cloud import CloudNode
from core.edge import EdgeNode
from core.network import NetworkModel
from core.pipeline import Pipeline
from core.queue import ProcessingQueue
from core.types import Location, ProfileCache
from evaluation.metrics import MetricsCollector
from strategies.admission import DropTail
from strategies.dispatch import (
    AdaptiveDeadlineDispatch,
    OneStepOptimalDispatch,
    FixedEdgeDispatch,
    FixedOffloadDispatch,
)
from tests.conftest import make_config


def _build_pipeline(
    config: SimulationConfig | None = None,
    arrival_times: list[float] | None = None,
    dispatcher: object | None = None,
    admission: object | None = None,
    n_profile_samples: int = 100,
) -> Pipeline:
    config = config or make_config()
    profile = ProfileCache.mock(n_samples=n_profile_samples, seed=config.seed)
    net = NetworkModel(
        base_rtt=config.network_base_rtt,
        jitter_std=config.network_jitter_std,
        seed=config.seed,
    )
    return Pipeline(
        profile=profile,
        queue=ProcessingQueue(config.queue_capacity),
        edge=EdgeNode(model_load_time=config.edge_model_load_time),
        cloud=CloudNode(capacity=config.cloud_capacity, network=net),
        admission=admission or DropTail(),
        dispatcher=dispatcher or FixedEdgeDispatch("full"),
        arrival_times=tuple(arrival_times or [0.0]),
        available_models=profile.available_models,
        metrics=MetricsCollector(),
        config=config,
    )


class TestPipelineSingleSample:
    def test_single_sample_flows_through(self) -> None:
        pipeline = _build_pipeline(arrival_times=[0.0])
        summary = pipeline.run()

        assert summary.total_arrivals == 1
        assert summary.total_processed == 1
        assert summary.total_dropped == 0

    def test_single_sample_accuracy_from_profile(self) -> None:
        pipeline = _build_pipeline(arrival_times=[0.0])
        summary = pipeline.run()

        assert summary.accuracy in (0.0, 1.0)

    def test_single_sample_latency_positive(self) -> None:
        pipeline = _build_pipeline(arrival_times=[0.0])
        summary = pipeline.run()

        assert summary.avg_latency > 0

    def test_single_sample_data_rate_positive(self) -> None:
        pipeline = _build_pipeline(arrival_times=[0.0])
        summary = pipeline.run()

        assert summary.avg_data_rate > 0


class TestPipelineQueueBehavior:
    def test_queue_fills_under_load(self) -> None:
        config = make_config(queue_capacity=5)
        pipeline = _build_pipeline(
            config=config,
            arrival_times=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        summary = pipeline.run()

        assert summary.total_dropped >= 1

    def test_expired_samples_dropped_from_queue(self) -> None:
        config = make_config(deadline_budget=0.01)
        pipeline = _build_pipeline(
            config=config,
            arrival_times=[0.0, 0.005],
        )
        summary = pipeline.run()

        assert summary.total_arrivals == 2
        assert "expired_in_queue" in summary.drops_by_reason

    def test_processed_late_samples_count_as_deadline_misses(self) -> None:
        config = make_config(deadline_budget=0.05)
        pipeline = _build_pipeline(
            config=config,
            arrival_times=[0.0],
            dispatcher=FixedEdgeDispatch("full"),
        )
        summary = pipeline.run()

        assert summary.total_processed == 1
        assert summary.deadline_miss_rate == 1.0


class TestPipelineEdgeCloud:
    def test_edge_and_cloud_process_in_parallel(self) -> None:
        pipeline = _build_pipeline(
            dispatcher=FixedOffloadDispatch("full"),
            arrival_times=[0.0, 0.0],
        )
        summary = pipeline.run()

        assert summary.total_processed == 2
        edge_infs = [r for r in pipeline._metrics.inferences if r.location == Location.EDGE]
        cloud_infs = [r for r in pipeline._metrics.inferences if r.location == Location.CLOUD]
        assert len(edge_infs) == 1
        assert len(cloud_infs) == 1

    def test_cloud_includes_network_latency(self) -> None:
        config = make_config(network_base_rtt=0.2, network_jitter_std=0.0)
        pipeline = _build_pipeline(
            config=config,
            dispatcher=FixedOffloadDispatch("full"),
            arrival_times=[0.0, 0.0],
        )
        pipeline.run()

        cloud_infs = [r for r in pipeline._metrics.inferences if r.location == Location.CLOUD]
        assert len(cloud_infs) == 1
        assert cloud_infs[0].processing_time > 0.2


class TestPipelineOneStepOptimal:
    def test_uses_both_devices_under_burst_load(self) -> None:
        config = make_config(network_base_rtt=0.0, network_jitter_std=0.0)
        pipeline = _build_pipeline(
            config=config,
            dispatcher=OneStepOptimalDispatch(),
            arrival_times=[0.0, 0.0, 0.0],
        )
        summary = pipeline.run()

        assert summary.total_processed == 3
        edge_infs = [r for r in pipeline._metrics.inferences if r.location == Location.EDGE]
        cloud_infs = [r for r in pipeline._metrics.inferences if r.location == Location.CLOUD]
        assert edge_infs
        assert cloud_infs


class TestPipelineDeadlineAwareAdaptive:
    def test_filters_infeasible_adaptive_dispatches(self) -> None:
        config = make_config(deadline_budget=0.001, network_base_rtt=0.0, network_jitter_std=0.0)
        pipeline = _build_pipeline(
            config=config,
            dispatcher=AdaptiveDeadlineDispatch(),
            arrival_times=[0.0],
        )
        summary = pipeline.run()

        assert summary.total_processed == 0
        assert summary.total_dropped == 1
        assert "dispatch_drop" in summary.drops_by_reason


class TestPipelineModelSwitching:
    def test_model_switch_recorded(self) -> None:
        pipeline = _build_pipeline(arrival_times=[0.0])
        summary = pipeline.run()

        assert summary.model_switch_count == 1
        assert summary.total_model_load_time > 0

    def test_no_switch_when_same_model(self) -> None:
        pipeline = _build_pipeline(arrival_times=[0.0, 1.0])
        summary = pipeline.run()

        assert summary.model_switch_count == 1


class TestPipelineDeterminism:
    def test_same_seed_same_results(self) -> None:
        config = make_config(seed=123, network_jitter_std=0.0)
        times = [0.1 * i for i in range(20)]

        p1 = _build_pipeline(
            config=config,
            arrival_times=times,
            dispatcher=FixedOffloadDispatch("full"),
        )
        s1 = p1.run()

        p2 = _build_pipeline(
            config=config,
            arrival_times=times,
            dispatcher=FixedOffloadDispatch("full"),
        )
        s2 = p2.run()

        assert s1.total_processed == s2.total_processed
        assert s1.accuracy == s2.accuracy
        assert s1.avg_data_rate == s2.avg_data_rate
        assert s1.avg_latency == s2.avg_latency
        assert s1.avg_aoi == s2.avg_aoi

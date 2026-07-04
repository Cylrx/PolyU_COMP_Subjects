from __future__ import annotations

from core.types import Location, ModelVariant
from strategies.dispatch import (
    AdaptiveDeadlineDispatch,
    AdaptiveQueueDispatch,
    OneStepOptimalDispatch,
    MultiStepOptimalDispatch,
)
from strategies.protocols import DispatchAction, DispatchContext
from tests.conftest import make_sample


def _context(
    queue: tuple,
    models: tuple[ModelVariant, ...],
    *,
    current_time: float = 0.0,
    queue_capacity: int | None = None,
    edge_available: bool = True,
    edge_next_available_time: float = 0.0,
    edge_loaded_model: str | None = None,
    edge_model_load_time: float = 0.05,
    cloud_available: bool = False,
    cloud_next_available_time: float = 0.0,
    expected_cloud_rtt: float = 0.1,
) -> DispatchContext:
    return DispatchContext(
        current_time=current_time,
        queue_capacity=queue_capacity if queue_capacity is not None else max(1, len(queue)),
        queued_samples=queue,
        available_models=models,
        edge_available=edge_available,
        edge_next_available_time=edge_next_available_time,
        edge_loaded_model=edge_loaded_model,
        edge_model_load_time=edge_model_load_time,
        cloud_available=cloud_available,
        cloud_next_available_time=cloud_next_available_time,
        expected_cloud_rtt=expected_cloud_rtt,
    )


class TestOneStepOptimalDispatch:
    def test_prefers_more_accurate_model_when_backlog_is_small(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.120, 0.090, 1),
            ModelVariant("fast", 0.90, 0.040, 0.030, 1),
        )
        queue = (make_sample(id=0, deadline=1.0),)
        context = _context(
            queue,
            models,
            queue_capacity=10,
            cloud_available=False,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.EDGE
        assert decision.model_name == "full"

    def test_switches_to_faster_model_when_backlog_is_large(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.120, 0.090, 1),
            ModelVariant("fast", 0.90, 0.040, 0.030, 1),
        )
        queue = tuple(make_sample(id=i, deadline=1.0) for i in range(8))
        context = _context(
            queue,
            models,
            queue_capacity=10,
            cloud_available=False,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.EDGE
        assert decision.model_name == "fast"

    def test_accounts_for_edge_switch_cost(self) -> None:
        models = (
            ModelVariant("accurate", 0.97, 0.120, 0.090, 1),
            ModelVariant("compact", 0.96, 0.060, 0.040, 1),
        )
        queue = tuple(make_sample(id=i, deadline=1.0) for i in range(4))
        context = _context(
            queue,
            models,
            queue_capacity=4,
            edge_loaded_model="accurate",
            cloud_available=False,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.EDGE
        assert decision.model_name == "accurate"

    def test_jointly_arbitrates_between_edge_and_cloud(self) -> None:
        models = (
            ModelVariant("full", 0.95, 0.200, 0.020, 1),
            ModelVariant("fast", 0.90, 0.100, 0.015, 1),
        )
        queue = tuple(make_sample(id=i, deadline=1.0) for i in range(3))
        context = _context(
            queue,
            models,
            queue_capacity=6,
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.CLOUD
        assert decision.model_name == "full"

    def test_uses_cloud_when_edge_is_busy_and_cloud_is_profitable(self) -> None:
        models = (
            ModelVariant("full", 0.95, 0.200, 0.020, 1),
            ModelVariant("fast", 0.90, 0.100, 0.015, 1),
        )
        queue = (make_sample(id=0, deadline=0.200),)
        context = _context(
            queue,
            models,
            queue_capacity=4,
            edge_available=False,
            edge_next_available_time=0.500,
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.CLOUD

    def test_waits_when_only_busy_device_can_still_meet_deadline(self) -> None:
        models = (
            ModelVariant("full", 0.90, 0.050, 0.200, 1),
            ModelVariant("fast", 0.80, 0.030, 0.180, 1),
        )
        queue = tuple(make_sample(id=i, deadline=0.100) for i in range(8))
        context = _context(
            queue,
            models,
            queue_capacity=10,
            edge_available=False,
            edge_next_available_time=0.010,
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.WAIT

    def test_drops_when_no_current_or_future_action_is_feasible(self) -> None:
        models = (
            ModelVariant("full", 0.90, 0.050, 0.200, 1),
            ModelVariant("fast", 0.80, 0.030, 0.180, 1),
        )
        queue = tuple(make_sample(id=i, deadline=0.100) for i in range(8))
        context = _context(
            queue,
            models,
            queue_capacity=10,
            edge_available=False,
            edge_next_available_time=0.200,
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.DROP

    def test_dispatches_feasible_action_even_when_backlog_penalty_is_large(self) -> None:
        models = (
            ModelVariant("full", 0.95, 0.200, 0.180, 1),
            ModelVariant("fast", 0.85, 0.120, 0.110, 1),
        )
        queue = tuple(make_sample(id=i, deadline=0.500) for i in range(10))
        context = _context(
            queue,
            models,
            queue_capacity=10,
            edge_available=False,
            edge_next_available_time=1.000,
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = OneStepOptimalDispatch().decide(queue[0], context)

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.CLOUD

    def test_ignores_sample_identity_for_equal_deadline_inputs(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.120, 0.090, 1),
            ModelVariant("fast", 0.90, 0.040, 0.030, 1),
        )
        sample_a = make_sample(id=0, dataset_idx=0, label=0, deadline=1.0)
        sample_b = make_sample(id=99, dataset_idx=77, label=9, deadline=1.0)
        context = _context(
            (sample_a,),
            models,
            queue_capacity=10,
            cloud_available=False,
            expected_cloud_rtt=0.0,
        )

        decision_a = OneStepOptimalDispatch().decide(sample_a, context)
        decision_b = OneStepOptimalDispatch().decide(sample_b, context)

        assert decision_a == decision_b

    def test_tuning_knobs_can_favor_faster_or_more_accurate_models(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.120, 0.090, 1),
            ModelVariant("fast", 0.90, 0.040, 0.030, 1),
        )
        queue = tuple(make_sample(id=i, deadline=1.0) for i in range(5))
        context = _context(
            queue,
            models,
            queue_capacity=10,
            cloud_available=False,
            expected_cloud_rtt=0.0,
        )

        accurate = OneStepOptimalDispatch(
            value_scale_multiplier=2.0,
            backlog_weight=1.0,
        ).decide(queue[0], context)
        throughput = OneStepOptimalDispatch(
            value_scale_multiplier=1.0,
            backlog_weight=2.0,
        ).decide(queue[0], context)

        assert accurate.action is DispatchAction.DISPATCH
        assert accurate.model_name == "full"
        assert throughput.action is DispatchAction.DISPATCH
        assert throughput.model_name == "fast"

    def test_rejects_invalid_tuning_parameters(self) -> None:
        try:
            OneStepOptimalDispatch(value_scale_multiplier=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected non-positive value_scale_multiplier to raise ValueError"
            )

        try:
            OneStepOptimalDispatch(backlog_weight=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected non-positive backlog_weight to raise ValueError"
            )


class TestAdaptiveQueueDispatch:
    @staticmethod
    def _decide(
        queue_size: int,
        models: tuple[ModelVariant, ...],
        *,
        queue_capacity: int,
    ) -> str | None:
        queue = tuple(make_sample(id=i, deadline=1.0) for i in range(queue_size))
        sample = queue[0] if queue else make_sample(id=0, deadline=1.0)
        return AdaptiveQueueDispatch().decide(
            sample,
            _context(queue, models, queue_capacity=queue_capacity),
        ).model_name

    def test_uses_uniform_accuracy_bands(self) -> None:
        models = (
            ModelVariant("model_a", 0.99, 0.010, 0.010, 1),
            ModelVariant("model_b", 0.98, 0.010, 0.010, 1),
            ModelVariant("model_c", 0.97, 0.010, 0.010, 1),
            ModelVariant("model_d", 0.96, 0.010, 0.010, 1),
            ModelVariant("model_e", 0.95, 0.010, 0.010, 1),
        )

        assert self._decide(0, models, queue_capacity=5) == "model_a"
        assert self._decide(1, models, queue_capacity=5) == "model_b"
        assert self._decide(2, models, queue_capacity=5) == "model_c"
        assert self._decide(3, models, queue_capacity=5) == "model_d"
        assert self._decide(4, models, queue_capacity=5) == "model_e"

    def test_reaches_every_band_with_six_models(self) -> None:
        models = tuple(
            ModelVariant(f"model_{idx}", 1.0 - idx * 0.01, 0.010, 0.010, 1)
            for idx in range(6)
        )

        for queue_size in range(6):
            assert (
                self._decide(queue_size, models, queue_capacity=6)
                == f"model_{queue_size}"
            )

    def test_clamps_to_fastest_band_when_queue_pressure_exceeds_one(self) -> None:
        models = (
            ModelVariant("accurate", 0.99, 0.010, 0.010, 1),
            ModelVariant("balanced", 0.95, 0.010, 0.010, 1),
            ModelVariant("fast", 0.90, 0.010, 0.010, 1),
        )

        assert self._decide(10, models, queue_capacity=3) == "fast"

    def test_uses_single_model_when_only_one_is_available(self) -> None:
        models = (ModelVariant("only", 0.99, 0.010, 0.010, 1),)
        queue = tuple(make_sample(id=i, deadline=1.0) for i in range(8))

        decision = AdaptiveQueueDispatch().decide(
            queue[0],
            _context(queue, models, queue_capacity=4),
        )

        assert decision.action is DispatchAction.DISPATCH
        assert decision.model_name == "only"

    def test_breaks_accuracy_ties_by_name(self) -> None:
        models = (
            ModelVariant("beta", 0.95, 0.010, 0.010, 1),
            ModelVariant("alpha", 0.95, 0.010, 0.010, 1),
        )
        queue = (make_sample(id=0, deadline=1.0),)

        decision = AdaptiveQueueDispatch().decide(
            queue[0],
            _context(queue, models, queue_capacity=4),
        )

        assert decision.model_name == "alpha"


class TestAdaptiveDeadlineDispatch:
    def test_uses_cloud_when_edge_cannot_meet_deadline(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.200, 0.040, 1),
            ModelVariant("fast", 0.90, 0.050, 0.030, 1),
        )
        queue = (make_sample(id=0, deadline=0.100),)
        context = _context(
            queue,
            models,
            queue_capacity=10,
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = AdaptiveDeadlineDispatch().decide(
            queue[0],
            context,
        )

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.CLOUD
        assert decision.model_name == "full"

    def test_waits_for_busy_preferred_device_when_it_can_still_meet_deadline(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.040, 0.200, 1),
            ModelVariant("fast", 0.90, 0.050, 0.030, 1),
        )
        queue = (make_sample(id=0, deadline=0.100),)
        context = _context(
            queue,
            models,
            queue_capacity=10,
            edge_available=False,
            edge_next_available_time=0.020,
            edge_loaded_model="full",
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = AdaptiveDeadlineDispatch().decide(
            queue[0],
            context,
        )

        assert decision.action is DispatchAction.WAIT

    def test_drops_when_selected_model_is_infeasible_everywhere(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.200, 0.180, 1),
            ModelVariant("fast", 0.90, 0.050, 0.030, 1),
        )
        queue = (make_sample(id=0, deadline=0.100),)
        context = _context(
            queue,
            models,
            queue_capacity=10,
            cloud_available=True,
            expected_cloud_rtt=0.0,
        )

        decision = AdaptiveDeadlineDispatch().decide(
            queue[0],
            context,
        )

        assert decision.action is DispatchAction.DROP


class TestMultiStepOptimalDispatch:
    def test_drops_infeasible_head_to_preserve_backlog(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.100, 0.030, 1),
            ModelVariant("fast", 0.90, 0.050, 0.020, 1),
        )
        queue = (
            make_sample(id=0, deadline=0.02),
            make_sample(id=1, deadline=0.20),
        )
        context = _context(queue, models, cloud_available=False)

        decision = MultiStepOptimalDispatch(
            max_backlog_window=50,
            time_quantum=0.005,
        ).decide(queue[0], context)

        assert decision.action is DispatchAction.DROP

    def test_uses_cloud_when_edge_is_busy_and_cloud_can_meet_deadline(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.060, 0.010, 1),
            ModelVariant("fast", 0.90, 0.030, 0.008, 1),
        )
        queue = (make_sample(id=0, deadline=0.12),)
        context = _context(
            queue,
            models,
            edge_available=False,
            edge_next_available_time=0.20,
            cloud_available=True,
            expected_cloud_rtt=0.05,
        )

        decision = MultiStepOptimalDispatch(
            max_backlog_window=50,
            time_quantum=0.005,
        ).decide(queue[0], context)

        assert decision.action is DispatchAction.DISPATCH
        assert decision.location is Location.CLOUD

    def test_never_dispatches_to_busy_cloud_when_quantization_merges_ticks(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.060, 0.010, 1),
            ModelVariant("fast", 0.90, 0.030, 0.008, 1),
        )
        queue = (make_sample(id=0, arrival_time=7.894576516880505, deadline=8.394576516880505),)
        context = _context(
            queue,
            models,
            current_time=7.894576516880505,
            cloud_available=False,
            cloud_next_available_time=7.9195241334268145,
            expected_cloud_rtt=0.1,
        )

        decision = MultiStepOptimalDispatch(
            max_backlog_window=50,
            time_quantum=0.03,
        ).decide(queue[0], context)

        assert not (
            decision.action is DispatchAction.DISPATCH
            and decision.location is Location.CLOUD
        )

    def test_service_time_penalty_can_favor_faster_edge_dispatch(self) -> None:
        models = (
            ModelVariant("full", 0.99, 0.200, 0.010, 1),
            ModelVariant("fast", 0.97, 0.010, 0.008, 1),
        )
        queue = (make_sample(id=0, deadline=0.12),)
        context = _context(
            queue,
            models,
            edge_model_load_time=0.0,
            cloud_available=True,
            expected_cloud_rtt=0.050,
        )

        baseline_decision = MultiStepOptimalDispatch(
            max_backlog_window=50,
            time_quantum=0.005,
            service_time_penalty=0.0,
        ).decide(queue[0], context)
        penalized_decision = MultiStepOptimalDispatch(
            max_backlog_window=50,
            time_quantum=0.005,
            service_time_penalty=0.5,
        ).decide(queue[0], context)

        assert baseline_decision.action is DispatchAction.DISPATCH
        assert baseline_decision.location is Location.CLOUD
        assert baseline_decision.model_name == "full"
        assert penalized_decision.action is DispatchAction.DISPATCH
        assert penalized_decision.location is Location.EDGE
        assert penalized_decision.model_name == "fast"

    def test_rejects_invalid_quantization_parameters(self) -> None:
        try:
            MultiStepOptimalDispatch(max_backlog_window=0)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid backlog window to raise ValueError")

        try:
            MultiStepOptimalDispatch(time_quantum=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid time quantum to raise ValueError")

        try:
            MultiStepOptimalDispatch(service_time_penalty=-0.1)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected negative service_time_penalty to raise ValueError"
            )

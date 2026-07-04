"""Unified dispatch strategies.

Each strategy chooses one of three actions for the queue head:
dispatch to (model, device), wait, or drop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import median

from core.types import Location, ModelVariant
from strategies.protocols import DispatchAction, DispatchContext, DispatchDecision


def _dispatch(model_name: str, location: Location) -> DispatchDecision:
    return DispatchDecision(
        action=DispatchAction.DISPATCH,
        model_name=model_name,
        location=location,
    )


def _wait() -> DispatchDecision:
    return DispatchDecision(action=DispatchAction.WAIT)


def _drop() -> DispatchDecision:
    return DispatchDecision(action=DispatchAction.DROP)


def _select_named_model(
    available_models: tuple[ModelVariant, ...],
    model_name: str,
) -> ModelVariant:
    return next(model for model in available_models if model.name == model_name)


def _sorted_models(
    available_models: tuple[ModelVariant, ...],
) -> tuple[ModelVariant, ...]:
    return tuple(sorted(available_models, key=lambda model: model.name))


def _models_by_accuracy_desc(
    available_models: tuple[ModelVariant, ...],
) -> tuple[ModelVariant, ...]:
    return tuple(
        sorted(
            available_models,
            key=lambda model: (-model.accuracy, model.name),
        )
    )


def _queue_pressure(context: DispatchContext) -> float:
    if context.queue_capacity <= 0:
        return 0.0
    return len(context.queued_samples) / context.queue_capacity


def _queue_backlog(context: DispatchContext) -> int:
    return len(context.queued_samples)


def _adaptive_band_index(queue_pressure: float, model_count: int) -> int:
    if model_count <= 1:
        return 0
    return min(int(queue_pressure * model_count), model_count - 1)


def _adaptive_model_name(context: DispatchContext) -> str:
    ranked_models = _models_by_accuracy_desc(context.available_models)
    return ranked_models[
        _adaptive_band_index(_queue_pressure(context), len(ranked_models))
    ].name


def _edge_service_time(
    model: ModelVariant,
    loaded_model: str | None,
    load_time: float,
) -> float:
    switch_time = load_time if loaded_model != model.name else 0.0
    return model.edge_avg_latency + switch_time


def _cloud_service_time(
    model: ModelVariant,
    expected_rtt: float,
) -> float:
    return model.cloud_avg_latency + expected_rtt


def _edge_finish_time_for_model(
    context: DispatchContext,
    model: ModelVariant,
    *,
    start_time: float | None = None,
) -> float:
    actual_start = context.current_time if start_time is None else start_time
    return actual_start + _edge_service_time(
        model,
        context.edge_loaded_model,
        context.edge_model_load_time,
    )


def _cloud_finish_time_for_model(
    context: DispatchContext,
    model: ModelVariant,
    *,
    start_time: float | None = None,
) -> float:
    actual_start = context.current_time if start_time is None else start_time
    return actual_start + _cloud_service_time(
        model,
        context.expected_cloud_rtt,
    )


def _current_deadline_feasible_dispatch_for_model(
    sample: object,
    context: DispatchContext,
    model: ModelVariant,
) -> DispatchDecision | None:
    if context.edge_available:
        if _edge_finish_time_for_model(context, model) <= sample.deadline:
            return _dispatch(model.name, Location.EDGE)
    if context.cloud_available:
        if _cloud_finish_time_for_model(context, model) <= sample.deadline:
            return _dispatch(model.name, Location.CLOUD)
    return None


def _future_deadline_feasible_dispatch_exists_for_model(
    sample: object,
    context: DispatchContext,
    model: ModelVariant,
) -> bool:
    if not context.edge_available:
        if (
            _edge_finish_time_for_model(
                context,
                model,
                start_time=context.edge_next_available_time,
            )
            <= sample.deadline
        ):
            return True

    if not context.cloud_available:
        if (
            _cloud_finish_time_for_model(
                context,
                model,
                start_time=context.cloud_next_available_time,
            )
            <= sample.deadline
        ):
            return True

    return False


def _non_dominated_models(
    models: tuple[ModelVariant, ...],
    key: Callable[[ModelVariant], float],
) -> tuple[ModelVariant, ...]:
    frontier: list[ModelVariant] = []
    for candidate in models:
        candidate_time = key(candidate)
        dominated = False
        for other in models:
            if other.name == candidate.name:
                continue
            other_time = key(other)
            if (
                other.accuracy >= candidate.accuracy
                and other_time <= candidate_time
                and (
                    other.accuracy > candidate.accuracy
                    or other_time < candidate_time
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda model: (-model.accuracy, model.name)))


def _positive_switch_threshold(
    slower_model: ModelVariant,
    faster_model: ModelVariant,
    key: Callable[[ModelVariant], float],
) -> float | None:
    utility_gap = slower_model.accuracy - faster_model.accuracy
    service_gap = key(slower_model) - key(faster_model)
    if utility_gap <= 0.0 or service_gap <= 0.0:
        return None
    return utility_gap / service_gap


def _effective_frontier_models(
    models: tuple[ModelVariant, ...],
    key: Callable[[ModelVariant], float],
) -> tuple[ModelVariant, ...]:
    frontier = tuple(
        sorted(
            _non_dominated_models(models, key),
            key=lambda model: (key(model), model.accuracy, model.name),
        )
    )
    if len(frontier) <= 2:
        return frontier

    envelope: list[ModelVariant] = []
    for candidate in reversed(frontier):
        while len(envelope) >= 2:
            left = _positive_switch_threshold(envelope[-2], envelope[-1], key)
            right = _positive_switch_threshold(envelope[-1], candidate, key)
            if (
                left is not None
                and right is not None
                and left >= right - 1e-12
            ):
                envelope.pop()
                continue
            break
        envelope.append(candidate)

    return tuple(reversed(envelope))


def _frontier_thresholds(
    models: tuple[ModelVariant, ...],
    key: Callable[[ModelVariant], float],
) -> tuple[float, ...]:
    slow_to_fast = tuple(reversed(_effective_frontier_models(models, key)))
    thresholds: list[float] = []
    for idx in range(len(slow_to_fast) - 1):
        threshold = _positive_switch_threshold(
            slow_to_fast[idx],
            slow_to_fast[idx + 1],
            key,
        )
        if threshold is not None and threshold > 0.0:
            thresholds.append(threshold)
    return tuple(thresholds)


@lru_cache(maxsize=None)
def _auto_tuned_value_scale(
    models: tuple[ModelVariant, ...],
    queue_capacity: int,
    expected_cloud_rtt: float,
) -> float:
    edge_thresholds = _frontier_thresholds(
        models,
        key=lambda model: model.edge_avg_latency,
    )
    cloud_thresholds = _frontier_thresholds(
        models,
        key=lambda model: model.cloud_avg_latency + expected_cloud_rtt,
    )
    positive_thresholds = tuple(
        threshold
        for threshold in (*edge_thresholds, *cloud_thresholds)
        if threshold > 0.0
    )

    q_ref = max(1.0, queue_capacity / 2.0)
    if not positive_thresholds:
        return q_ref

    p_ref = median(positive_thresholds)
    if p_ref <= 0.0:
        return q_ref
    return q_ref / p_ref


def _dispatch_score(
    backlog: int,
    backlog_weight: float,
    value_scale: float,
    utility: float,
    service_time: float,
) -> float:
    return value_scale * utility - backlog_weight * float(backlog) * service_time


@dataclass(frozen=True)
class _ScoredDispatchCandidate:
    decision: DispatchDecision
    score: float
    finish_time: float
    utility: float


def _is_better_candidate(
    candidate: _ScoredDispatchCandidate,
    incumbent: _ScoredDispatchCandidate | None,
) -> bool:
    if incumbent is None:
        return True

    candidate_priority = 1 if candidate.decision.location is Location.EDGE else 0
    incumbent_priority = 1 if incumbent.decision.location is Location.EDGE else 0

    candidate_key = (
        candidate.score,
        -candidate.finish_time,
        candidate.utility,
        candidate_priority,
    )
    incumbent_key = (
        incumbent.score,
        -incumbent.finish_time,
        incumbent.utility,
        incumbent_priority,
    )
    if candidate_key != incumbent_key:
        return candidate_key > incumbent_key

    candidate_name = candidate.decision.model_name or ""
    incumbent_name = incumbent.decision.model_name or ""
    return candidate_name < incumbent_name


def _best_current_edge_candidate(
    sample: object,
    context: DispatchContext,
    models: tuple[ModelVariant, ...],
    backlog: int,
    backlog_weight: float,
    value_scale: float,
) -> _ScoredDispatchCandidate | None:
    if not context.edge_available:
        return None

    edge_models = _effective_frontier_models(
        models,
        key=lambda model: _edge_service_time(
            model,
            context.edge_loaded_model,
            context.edge_model_load_time,
        ),
    )

    best: _ScoredDispatchCandidate | None = None
    for model in edge_models:
        service_time = _edge_service_time(
            model,
            context.edge_loaded_model,
            context.edge_model_load_time,
        )
        finish_time = context.current_time + service_time
        if finish_time > sample.deadline:
            continue

        candidate = _ScoredDispatchCandidate(
            decision=_dispatch(model.name, Location.EDGE),
            score=_dispatch_score(
                backlog=backlog,
                backlog_weight=backlog_weight,
                value_scale=value_scale,
                utility=model.accuracy,
                service_time=service_time,
            ),
            finish_time=finish_time,
            utility=model.accuracy,
        )
        if _is_better_candidate(candidate, best):
            best = candidate

    return best


def _best_current_cloud_candidate(
    sample: object,
    context: DispatchContext,
    models: tuple[ModelVariant, ...],
    backlog: int,
    backlog_weight: float,
    value_scale: float,
) -> _ScoredDispatchCandidate | None:
    if not context.cloud_available:
        return None

    cloud_models = _effective_frontier_models(
        models,
        key=lambda model: _cloud_service_time(
            model,
            context.expected_cloud_rtt,
        ),
    )

    best: _ScoredDispatchCandidate | None = None
    for model in cloud_models:
        service_time = _cloud_service_time(
            model,
            context.expected_cloud_rtt,
        )
        finish_time = context.current_time + service_time
        if finish_time > sample.deadline:
            continue

        candidate = _ScoredDispatchCandidate(
            decision=_dispatch(model.name, Location.CLOUD),
            score=_dispatch_score(
                backlog=backlog,
                backlog_weight=backlog_weight,
                value_scale=value_scale,
                utility=model.accuracy,
                service_time=service_time,
            ),
            finish_time=finish_time,
            utility=model.accuracy,
        )
        if _is_better_candidate(candidate, best):
            best = candidate

    return best


def _has_future_feasible_busy_dispatch(
    sample: object,
    context: DispatchContext,
    models: tuple[ModelVariant, ...],
) -> bool:
    if not context.edge_available:
        fastest_edge_service = min(
            _edge_service_time(
                model,
                context.edge_loaded_model,
                context.edge_model_load_time,
            )
            for model in models
        )
        if context.edge_next_available_time + fastest_edge_service <= sample.deadline:
            return True

    if not context.cloud_available:
        fastest_cloud_service = min(
            _cloud_service_time(
                model,
                context.expected_cloud_rtt,
            )
            for model in models
        )
        if context.cloud_next_available_time + fastest_cloud_service <= sample.deadline:
            return True

    return False


class _ProgressReportingDispatch:
    """Small helper for strategies that report the current arrival index."""

    def __init__(self) -> None:
        self._progress_reporter: Callable[[int], None] | None = None

    def set_progress_reporter(
        self,
        reporter: Callable[[int], None] | None,
    ) -> None:
        self._progress_reporter = reporter

    def _report_progress(self, sample: object) -> None:
        if self._progress_reporter is not None and hasattr(sample, "id"):
            self._progress_reporter(sample.id + 1)


class FixedEdgeDispatch(_ProgressReportingDispatch):
    """Always use one model on edge, never offload."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name

    def decide(
        self,
        sample: object,
        context: DispatchContext,
    ) -> DispatchDecision:
        self._report_progress(sample)
        if context.edge_available:
            model = _select_named_model(context.available_models, self._model_name)
            return _dispatch(model.name, Location.EDGE)
        return _wait()


class FixedOffloadDispatch(_ProgressReportingDispatch):
    """Use one model everywhere; prefer edge, then cloud."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name

    def decide(
        self,
        sample: object,
        context: DispatchContext,
    ) -> DispatchDecision:
        self._report_progress(sample)
        model = _select_named_model(context.available_models, self._model_name)
        if context.edge_available:
            return _dispatch(model.name, Location.EDGE)
        if context.cloud_available:
            return _dispatch(model.name, Location.CLOUD)
        return _wait()


class AdaptiveQueueDispatch(_ProgressReportingDispatch):
    """Select models from uniform queue-pressure bands; prefer edge, then cloud."""

    def __init__(self) -> None:
        super().__init__()

    def _select_model_name(self, context: DispatchContext) -> str:
        return _adaptive_model_name(context)

    def decide(
        self,
        sample: object,
        context: DispatchContext,
    ) -> DispatchDecision:
        self._report_progress(sample)
        model_name = self._select_model_name(context)

        if context.edge_available:
            return _dispatch(model_name, Location.EDGE)
        if context.cloud_available:
            return _dispatch(model_name, Location.CLOUD)
        return _wait()


class AdaptiveDeadlineDispatch(AdaptiveQueueDispatch):
    """Queue-adaptive dispatch with hard deadline feasibility checks."""

    def decide(
        self,
        sample: object,
        context: DispatchContext,
    ) -> DispatchDecision:
        self._report_progress(sample)
        model = _select_named_model(
            context.available_models,
            self._select_model_name(context),
        )

        current_dispatch = _current_deadline_feasible_dispatch_for_model(
            sample,
            context,
            model,
        )
        if current_dispatch is not None:
            return current_dispatch

        if _future_deadline_feasible_dispatch_exists_for_model(sample, context, model):
            return _wait()

        return _drop()


class OneStepOptimalDispatch(_ProgressReportingDispatch):
    """Single-queue DPP dispatch using model-level utility and service cost."""

    def __init__(
        self,
        *,
        value_scale_multiplier: float = 1.0,
        backlog_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if value_scale_multiplier <= 0.0:
            raise ValueError(
                "value_scale_multiplier must be positive, "
                f"got {value_scale_multiplier}"
            )
        if backlog_weight <= 0.0:
            raise ValueError(
                f"backlog_weight must be positive, got {backlog_weight}"
            )
        self._value_scale_multiplier = value_scale_multiplier
        self._backlog_weight = backlog_weight

    def decide(
        self,
        sample: object,
        context: DispatchContext,
    ) -> DispatchDecision:
        self._report_progress(sample)

        if not context.queued_samples:
            return _wait()

        models = _sorted_models(context.available_models)
        backlog = _queue_backlog(context)
        value_scale = _auto_tuned_value_scale(
            models,
            context.queue_capacity,
            context.expected_cloud_rtt,
        )
        value_scale *= self._value_scale_multiplier

        best_candidate: _ScoredDispatchCandidate | None = None
        for candidate in (
            _best_current_edge_candidate(
                sample,
                context,
                models,
                backlog,
                self._backlog_weight,
                value_scale,
            ),
            _best_current_cloud_candidate(
                sample,
                context,
                models,
                backlog,
                self._backlog_weight,
                value_scale,
            ),
        ):
            if candidate is not None and _is_better_candidate(candidate, best_candidate):
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate.decision

        if _has_future_feasible_busy_dispatch(sample, context, models):
            return _wait()

        return _drop()


class MultiStepOptimalDispatch(_ProgressReportingDispatch):
    """Exact backlog-only joint planning with first-action execution."""

    def __init__(
        self,
        max_backlog_window: int = 50,
        time_quantum: float = 0.005,
        service_time_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        if max_backlog_window <= 0:
            raise ValueError(
                f"max_backlog_window must be positive, got {max_backlog_window}"
            )
        if time_quantum <= 0:
            raise ValueError(f"time_quantum must be positive, got {time_quantum}")
        if service_time_penalty < 0.0:
            raise ValueError(
                "service_time_penalty must be non-negative, "
                f"got {service_time_penalty}"
            )
        self._max_backlog_window = max_backlog_window
        self._time_quantum = time_quantum
        self._service_time_penalty = service_time_penalty

    def _to_tick(self, value: float) -> int:
        return math.ceil(value / self._time_quantum - 1e-12)

    def _to_deadline_tick(self, value: float) -> int:
        return math.floor(value / self._time_quantum + 1e-12)

    def decide(
        self,
        sample: object,
        context: DispatchContext,
    ) -> DispatchDecision:
        self._report_progress(sample)

        samples = context.queued_samples[: self._max_backlog_window]
        if not samples:
            return _wait()

        models = _sorted_models(context.available_models)
        model_names = tuple(model.name for model in models)
        model_index = {name: idx for idx, name in enumerate(model_names)}
        cloud_frontier = _non_dominated_models(
            models,
            key=lambda model: _cloud_service_time(
                model,
                context.expected_cloud_rtt,
            ),
        )
        cloud_frontier_indices = tuple(
            model_index[model.name] for model in cloud_frontier
        )

        loaded_state_indices = (-1, *range(len(models)))
        edge_frontiers = {
            loaded_idx: _non_dominated_models(
                models,
                key=lambda model: _edge_service_time(
                    model,
                    None if loaded_idx < 0 else model_names[loaded_idx],
                    context.edge_model_load_time,
                ),
            )
            for loaded_idx in loaded_state_indices
        }
        edge_frontier_indices = {
            loaded_idx: tuple(model_index[model.name] for model in frontier)
            for loaded_idx, frontier in edge_frontiers.items()
        }
        edge_service_ticks = {
            loaded_idx: {
                model_idx: self._to_tick(
                    _edge_service_time(
                        models[model_idx],
                        None if loaded_idx < 0 else model_names[loaded_idx],
                        context.edge_model_load_time,
                    )
                )
                for model_idx in range(len(models))
            }
            for loaded_idx in loaded_state_indices
        }
        edge_rewards = {
            loaded_idx: {
                model_idx: (
                    models[model_idx].accuracy
                    - self._service_time_penalty
                    * _edge_service_time(
                        models[model_idx],
                        None if loaded_idx < 0 else model_names[loaded_idx],
                        context.edge_model_load_time,
                    )
                )
                for model_idx in range(len(models))
            }
            for loaded_idx in loaded_state_indices
        }
        cloud_service_ticks = {
            model_idx: self._to_tick(
                _cloud_service_time(
                    models[model_idx],
                    context.expected_cloud_rtt,
                )
            )
            for model_idx in range(len(models))
        }
        cloud_rewards = {
            model_idx: (
                models[model_idx].accuracy
                - self._service_time_penalty
                * _cloud_service_time(
                    models[model_idx],
                    context.expected_cloud_rtt,
                )
            )
            for model_idx in range(len(models))
        }
        deadline_ticks = tuple(
            self._to_deadline_tick(queued_sample.deadline) for queued_sample in samples
        )
        current_tick = self._to_tick(context.current_time)

        @lru_cache(maxsize=None)
        def value(
            index: int,
            current_tick: int,
            edge_ready_tick: int,
            edge_loaded_idx: int,
            cloud_ready_tick: int,
        ) -> float:
            while index < len(samples) and deadline_ticks[index] <= current_tick:
                index += 1

            if index >= len(samples):
                return 0.0

            edge_available = edge_ready_tick <= current_tick
            cloud_available = cloud_ready_tick <= current_tick

            if not edge_available and not cloud_available:
                return value(
                    index,
                    min(edge_ready_tick, cloud_ready_tick),
                    edge_ready_tick,
                    edge_loaded_idx,
                    cloud_ready_tick,
                )

            best = value(
                index + 1,
                current_tick,
                edge_ready_tick,
                edge_loaded_idx,
                cloud_ready_tick,
            )

            head_deadline_tick = deadline_ticks[index]

            if edge_available:
                for model_idx in edge_frontier_indices[edge_loaded_idx]:
                    finish_tick = current_tick + edge_service_ticks[edge_loaded_idx][
                        model_idx
                    ]
                    if finish_tick <= head_deadline_tick:
                        best = max(
                            best,
                            edge_rewards[edge_loaded_idx][model_idx]
                            + value(
                                index + 1,
                                current_tick,
                                finish_tick,
                                model_idx,
                                cloud_ready_tick,
                            ),
                        )

            if cloud_available:
                for model_idx in cloud_frontier_indices:
                    finish_tick = current_tick + cloud_service_ticks[model_idx]
                    if finish_tick <= head_deadline_tick:
                        best = max(
                            best,
                            cloud_rewards[model_idx]
                            + value(
                                index + 1,
                                current_tick,
                                edge_ready_tick,
                                edge_loaded_idx,
                                finish_tick,
                            ),
                        )

            next_event = min(
                (
                    ready_tick
                    for ready_tick in (edge_ready_tick, cloud_ready_tick)
                    if ready_tick > current_tick
                ),
                default=None,
            )
            if next_event is not None:
                best = max(
                    best,
                    value(
                        index,
                        next_event,
                        edge_ready_tick,
                        edge_loaded_idx,
                        cloud_ready_tick,
                    ),
                )

            return best

        edge_loaded_idx = (
            -1
            if context.edge_loaded_model is None
            else model_index[context.edge_loaded_model]
        )
        edge_ready_tick = (
            current_tick
            if context.edge_available
            else self._to_tick(context.edge_next_available_time)
        )
        cloud_ready_tick = (
            current_tick
            if context.cloud_available
            else self._to_tick(context.cloud_next_available_time)
        )
        exact_edge_available = context.edge_available
        exact_cloud_available = context.cloud_available

        actions: list[tuple[float, int, DispatchDecision]] = []

        drop_value = value(
            1,
            current_tick,
            edge_ready_tick,
            edge_loaded_idx,
            cloud_ready_tick,
        )
        actions.append((drop_value, 0, _drop()))

        # The first action must be legal in the real simulator state, even
        # though continuation values are computed on quantized time buckets.
        if exact_edge_available:
            for model_idx in edge_frontier_indices[edge_loaded_idx]:
                finish_tick = current_tick + edge_service_ticks[edge_loaded_idx][model_idx]
                if finish_tick <= deadline_ticks[0]:
                    total_value = edge_rewards[edge_loaded_idx][model_idx] + value(
                        1,
                        current_tick,
                        finish_tick,
                        model_idx,
                        cloud_ready_tick,
                    )
                    actions.append(
                        (
                            total_value,
                            3,
                            _dispatch(models[model_idx].name, Location.EDGE),
                        )
                    )

        if exact_cloud_available:
            for model_idx in cloud_frontier_indices:
                finish_tick = current_tick + cloud_service_ticks[model_idx]
                if finish_tick <= deadline_ticks[0]:
                    total_value = cloud_rewards[model_idx] + value(
                        1,
                        current_tick,
                        edge_ready_tick,
                        edge_loaded_idx,
                        finish_tick,
                    )
                    actions.append(
                        (
                            total_value,
                            2,
                            _dispatch(models[model_idx].name, Location.CLOUD),
                        )
                    )

        next_event = min(
            (
                ready_tick
                for ready_tick in (edge_ready_tick, cloud_ready_tick)
                if ready_tick > current_tick
            ),
            default=None,
        )
        if next_event is not None and deadline_ticks[0] > next_event:
            wait_value = value(
                0,
                next_event,
                edge_ready_tick,
                edge_loaded_idx,
                cloud_ready_tick,
            )
            actions.append((wait_value, 1, _wait()))

        _, _, best_decision = max(actions, key=lambda item: (item[0], item[1]))
        return best_decision

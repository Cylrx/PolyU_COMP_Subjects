from __future__ import annotations

import math
from dataclasses import dataclass

from core.types import Location


@dataclass(frozen=True)
class ArrivalRecord:
    sample_id: int
    time: float


@dataclass(frozen=True)
class DropRecord:
    sample_id: int
    time: float
    reason: str


@dataclass(frozen=True)
class InferenceRecord:
    sample_id: int
    prediction: int
    correct: bool
    model_name: str
    location: Location
    arrival_time: float
    deadline: float
    enqueue_time: float
    start_time: float
    end_time: float

    @property
    def e2e_latency(self) -> float:
        return self.end_time - self.arrival_time

    @property
    def queue_wait(self) -> float:
        return self.start_time - self.enqueue_time

    @property
    def processing_time(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class ModelSwitchRecord:
    time: float
    from_model: str | None
    to_model: str
    load_duration: float


@dataclass(frozen=True)
class QueueSnapshot:
    time: float
    size: int


@dataclass
class MetricsSummary:
    # Overall
    total_arrivals: int
    total_processed: int
    total_dropped: int
    drop_rate: float
    avg_data_rate: float

    # Latency
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float

    # Data freshness
    avg_aoi: float

    # Accuracy
    accuracy: float

    # Deadline
    deadline_miss_rate: float

    # Edge/Cloud distribution
    edge_utilization: float
    cloud_utilization: float
    offload_ratio: float

    # Per-model breakdown
    per_model_accuracy: dict[str, float]
    per_model_avg_latency: dict[str, float]
    per_model_count: dict[str, int]

    # Drop breakdown
    drops_by_reason: dict[str, int]

    # Model switching
    model_switch_count: int
    total_model_load_time: float

    # Time series
    queue_length_over_time: list[tuple[float, int]]

    # Per-interval time series (for line charts)
    data_rate_over_time: list[tuple[float, float]]
    accuracy_over_time: list[tuple[float, float]]
    deadline_misses_over_time: list[tuple[float, int]]

    # Raw per-sample latencies (for histograms)
    all_latencies: list[float]

    @classmethod
    def average(cls, summaries: list[MetricsSummary]) -> MetricsSummary:
        """Average multiple summaries from repeated experiment runs."""
        if len(summaries) == 1:
            return summaries[0]
        n = len(summaries)

        def _mean(values: list[float]) -> float:
            return sum(values) / n

        def _mean_int(values: list[int]) -> int:
            return round(sum(values) / n)

        def _mean_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
            keys = sorted({k for d in dicts for k in d})
            return {k: sum(d.get(k, 0.0) for d in dicts) / n for k in keys}

        def _mean_dict_int(dicts: list[dict[str, int]]) -> dict[str, int]:
            keys = sorted({k for d in dicts for k in d})
            return {k: round(sum(d.get(k, 0) for d in dicts) / n) for k in keys}

        def _mean_ts(
            all_series: list[list[tuple[float, float]]],
        ) -> list[tuple[float, float]]:
            if not all_series or not all_series[0]:
                return []
            length = min(len(s) for s in all_series)
            return [
                (all_series[0][i][0], _mean([s[i][1] for s in all_series]))
                for i in range(length)
            ]

        def _mean_ts_int(
            all_series: list[list[tuple[float, int]]],
        ) -> list[tuple[float, int]]:
            if not all_series or not all_series[0]:
                return []
            length = min(len(s) for s in all_series)
            return [
                (all_series[0][i][0], round(sum(s[i][1] for s in all_series) / n))
                for i in range(length)
            ]

        return cls(
            total_arrivals=_mean_int([s.total_arrivals for s in summaries]),
            total_processed=_mean_int([s.total_processed for s in summaries]),
            total_dropped=_mean_int([s.total_dropped for s in summaries]),
            drop_rate=_mean([s.drop_rate for s in summaries]),
            avg_data_rate=_mean([s.avg_data_rate for s in summaries]),
            avg_latency=_mean([s.avg_latency for s in summaries]),
            p50_latency=_mean([s.p50_latency for s in summaries]),
            p95_latency=_mean([s.p95_latency for s in summaries]),
            p99_latency=_mean([s.p99_latency for s in summaries]),
            avg_aoi=_mean([s.avg_aoi for s in summaries]),
            accuracy=_mean([s.accuracy for s in summaries]),
            deadline_miss_rate=_mean([s.deadline_miss_rate for s in summaries]),
            edge_utilization=_mean([s.edge_utilization for s in summaries]),
            cloud_utilization=_mean([s.cloud_utilization for s in summaries]),
            offload_ratio=_mean([s.offload_ratio for s in summaries]),
            per_model_accuracy=_mean_dict([s.per_model_accuracy for s in summaries]),
            per_model_avg_latency=_mean_dict([s.per_model_avg_latency for s in summaries]),
            per_model_count=_mean_dict_int([s.per_model_count for s in summaries]),
            drops_by_reason=_mean_dict_int([s.drops_by_reason for s in summaries]),
            model_switch_count=_mean_int([s.model_switch_count for s in summaries]),
            total_model_load_time=_mean([s.total_model_load_time for s in summaries]),
            queue_length_over_time=summaries[-1].queue_length_over_time,
            data_rate_over_time=_mean_ts([s.data_rate_over_time for s in summaries]),
            accuracy_over_time=_mean_ts([s.accuracy_over_time for s in summaries]),
            deadline_misses_over_time=_mean_ts_int(
                [s.deadline_misses_over_time for s in summaries],
            ),
            all_latencies=[lat for s in summaries for lat in s.all_latencies],
        )


class MetricsCollector:
    """Collects raw event logs during simulation, computes summaries lazily."""

    def __init__(self) -> None:
        self.arrivals: list[ArrivalRecord] = []
        self.drops: list[DropRecord] = []
        self.inferences: list[InferenceRecord] = []
        self.model_switches: list[ModelSwitchRecord] = []
        self.queue_snapshots: list[QueueSnapshot] = []
        self._enqueue_times: dict[int, float] = {}

    def record_arrival(self, sample_id: int, time: float) -> None:
        self.arrivals.append(ArrivalRecord(sample_id=sample_id, time=time))

    def record_drop(self, sample_id: int, time: float, reason: str) -> None:
        self.drops.append(DropRecord(sample_id=sample_id, time=time, reason=reason))

    def record_enqueue(self, sample_id: int, time: float, queue_size: int) -> None:
        self._enqueue_times[sample_id] = time
        self.queue_snapshots.append(QueueSnapshot(time=time, size=queue_size))

    def record_inference(
        self,
        sample_id: int,
        prediction: int,
        correct: bool,
        model_name: str,
        location: Location,
        arrival_time: float,
        deadline: float,
        start_time: float,
        end_time: float,
    ) -> None:
        enqueue_time = self._enqueue_times.get(sample_id, arrival_time)
        self.inferences.append(
            InferenceRecord(
                sample_id=sample_id,
                prediction=prediction,
                correct=correct,
                model_name=model_name,
                location=location,
                arrival_time=arrival_time,
                deadline=deadline,
                enqueue_time=enqueue_time,
                start_time=start_time,
                end_time=end_time,
            )
        )

    def record_model_switch(
        self,
        time: float,
        from_model: str | None,
        to_model: str,
        load_duration: float,
    ) -> None:
        self.model_switches.append(
            ModelSwitchRecord(
                time=time,
                from_model=from_model,
                to_model=to_model,
                load_duration=load_duration,
            )
        )

    def summary(self, duration: float, metric_interval: float = 5.0) -> MetricsSummary:
        total_arrivals = len(self.arrivals)
        total_processed = len(self.inferences)
        total_dropped = len(self.drops)

        latencies = [r.e2e_latency for r in self.inferences]
        sorted_lat = sorted(latencies) if latencies else [0.0]

        # Per-model breakdown
        model_correct: dict[str, list[bool]] = {}
        model_latencies: dict[str, list[float]] = {}
        model_count: dict[str, int] = {}
        edge_count = 0
        cloud_count = 0
        processed_deadline_misses = 0

        for r in self.inferences:
            model_correct.setdefault(r.model_name, []).append(r.correct)
            model_latencies.setdefault(r.model_name, []).append(r.e2e_latency)
            model_count[r.model_name] = model_count.get(r.model_name, 0) + 1
            if r.location == Location.EDGE:
                edge_count += 1
            else:
                cloud_count += 1
            if r.end_time > r.deadline:
                processed_deadline_misses += 1

        deadline_miss_drops = sum(
            1 for d in self.drops if d.reason in ("expired_on_arrival", "expired_in_queue")
        )
        total_considered = total_processed + deadline_miss_drops
        deadline_misses = processed_deadline_misses + deadline_miss_drops

        # Drop breakdown
        drops_by_reason: dict[str, int] = {}
        for d in self.drops:
            drops_by_reason[d.reason] = drops_by_reason.get(d.reason, 0) + 1

        data_rate_ts, accuracy_ts, deadline_miss_ts = _compute_interval_time_series(
            self.inferences, self.drops, duration, metric_interval,
        )

        return MetricsSummary(
            total_arrivals=total_arrivals,
            total_processed=total_processed,
            total_dropped=total_dropped,
            drop_rate=total_dropped / total_arrivals if total_arrivals else 0.0,
            avg_data_rate=_average_data_rate(total_processed, duration),
            avg_latency=_mean(latencies),
            p50_latency=_percentile(sorted_lat, 0.50),
            p95_latency=_percentile(sorted_lat, 0.95),
            p99_latency=_percentile(sorted_lat, 0.99),
            avg_aoi=_average_aoi(self.inferences, duration),
            accuracy=(
                sum(r.correct for r in self.inferences) / total_processed
                if total_processed
                else 0.0
            ),
            deadline_miss_rate=(
                deadline_misses / total_considered if total_considered else 0.0
            ),
            edge_utilization=edge_count / total_processed if total_processed else 0.0,
            cloud_utilization=cloud_count / total_processed if total_processed else 0.0,
            offload_ratio=cloud_count / total_processed if total_processed else 0.0,
            per_model_accuracy={
                m: sum(c) / len(c) for m, c in model_correct.items()
            },
            per_model_avg_latency={
                m: _mean(lats) for m, lats in model_latencies.items()
            },
            per_model_count=model_count,
            drops_by_reason=drops_by_reason,
            model_switch_count=len(self.model_switches),
            total_model_load_time=sum(s.load_duration for s in self.model_switches),
            queue_length_over_time=[
                (s.time, s.size) for s in self.queue_snapshots
            ],
            data_rate_over_time=data_rate_ts,
            accuracy_over_time=accuracy_ts,
            deadline_misses_over_time=deadline_miss_ts,
            all_latencies=latencies,
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _average_data_rate(total_processed: int, duration: float) -> float:
    return total_processed / duration if duration > 0 else 0.0


_DEADLINE_DROP_REASONS = frozenset({"expired_on_arrival", "expired_in_queue"})


def _compute_interval_time_series(
    inferences: list[InferenceRecord],
    drops: list[DropRecord],
    duration: float,
    interval: float,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, int]],
]:
    """Compute per-interval data rate, accuracy, and deadline miss count.

    Returns three aligned time series whose X values are the right edge of
    each interval.  Empty-interval accuracy carries forward the last known
    value (initialised to 0.0).
    """
    n_intervals = math.ceil(duration / interval) if duration > 0 else 0

    data_rate: list[tuple[float, float]] = []
    accuracy: list[tuple[float, float]] = []
    deadline_misses: list[tuple[float, int]] = []

    last_accuracy = 0.0

    for i in range(n_intervals):
        t_start = i * interval
        t_end = min((i + 1) * interval, duration)
        length = t_end - t_start

        # Inferences completing in this interval.
        window_infs = [
            r for r in inferences if t_start <= r.end_time < t_end
        ]
        count = len(window_infs)

        # Data rate.
        data_rate.append((t_end, count / length if length > 0 else 0.0))

        # Accuracy (carry forward on empty interval).
        if count > 0:
            last_accuracy = sum(r.correct for r in window_infs) / count
        accuracy.append((t_end, last_accuracy))

        # Deadline misses: late completions + expiration drops.
        late = sum(1 for r in window_infs if r.end_time > r.deadline)
        expired = sum(
            1 for d in drops
            if d.reason in _DEADLINE_DROP_REASONS and t_start <= d.time < t_end
        )
        deadline_misses.append((t_end, late + expired))

    return data_rate, accuracy, deadline_misses


def _average_aoi(inferences: list[InferenceRecord], duration: float) -> float:
    """Compute average AoI over [first completion, duration].

    This simulator currently treats arrival time as the sample generation time.
    AoI is undefined until the first inference completes within the configured
    duration, so this returns NaN when no completion occurs in that window.
    """
    completions = sorted(
        (record.end_time, record.arrival_time)
        for record in inferences
        if record.end_time <= duration
    )
    if not completions:
        return math.nan

    current_time = completions[0][0]
    if duration <= current_time:
        return math.nan

    freshest_generation = completions[0][1]
    current_index = 1
    while current_index < len(completions) and completions[current_index][0] == current_time:
        freshest_generation = max(freshest_generation, completions[current_index][1])
        current_index += 1

    area = 0.0
    while current_index < len(completions):
        next_time = completions[current_index][0]
        left_height = current_time - freshest_generation
        right_height = next_time - freshest_generation
        area += (left_height + right_height) * (next_time - current_time) / 2.0

        while current_index < len(completions) and completions[current_index][0] == next_time:
            freshest_generation = max(freshest_generation, completions[current_index][1])
            current_index += 1
        current_time = next_time

    left_height = current_time - freshest_generation
    right_height = duration - freshest_generation
    area += (left_height + right_height) * (duration - current_time) / 2.0

    return area / (duration - completions[0][0])


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(p * (len(sorted_values) - 1))
    return sorted_values[idx]

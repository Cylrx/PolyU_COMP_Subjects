from __future__ import annotations

import math

import pytest

from core.types import Location
from evaluation.metrics import MetricsCollector


def _record_inference(
    collector: MetricsCollector,
    *,
    sample_id: int,
    arrival_time: float,
    end_time: float,
    start_time: float | None = None,
) -> None:
    collector.record_inference(
        sample_id=sample_id,
        prediction=0,
        correct=True,
        model_name="full",
        location=Location.EDGE,
        arrival_time=arrival_time,
        deadline=100.0,
        start_time=arrival_time if start_time is None else start_time,
        end_time=end_time,
    )


class TestMetricsAoI:
    def test_avg_aoi_uses_exact_trapezoid_integration(self) -> None:
        collector = MetricsCollector()
        _record_inference(collector, sample_id=0, arrival_time=1.0, end_time=4.0)
        _record_inference(collector, sample_id=1, arrival_time=5.0, end_time=8.0)

        summary = collector.summary(duration=10.0)

        assert summary.avg_aoi == pytest.approx(14.0 / 3.0)

    def test_avg_aoi_uses_freshest_delivered_information(self) -> None:
        collector = MetricsCollector()
        _record_inference(collector, sample_id=0, arrival_time=5.0, end_time=8.0)
        _record_inference(collector, sample_id=1, arrival_time=1.0, end_time=9.0)

        summary = collector.summary(duration=10.0)

        assert summary.avg_aoi == pytest.approx(4.0)

    def test_avg_aoi_groups_same_time_completions_before_integrating(self) -> None:
        collector = MetricsCollector()
        _record_inference(collector, sample_id=0, arrival_time=2.0, end_time=8.0)
        _record_inference(collector, sample_id=1, arrival_time=5.0, end_time=8.0)

        summary = collector.summary(duration=10.0)

        assert summary.avg_aoi == pytest.approx(4.0)

    def test_avg_aoi_is_nan_when_no_completion_occurs_in_window(self) -> None:
        collector = MetricsCollector()
        _record_inference(collector, sample_id=0, arrival_time=1.0, end_time=12.0)

        summary = collector.summary(duration=10.0)

        assert math.isnan(summary.avg_aoi)


class TestMetricsDataRate:
    def test_avg_data_rate_uses_processed_samples_over_duration(self) -> None:
        collector = MetricsCollector()
        _record_inference(collector, sample_id=0, arrival_time=0.0, end_time=1.0)
        _record_inference(collector, sample_id=1, arrival_time=1.0, end_time=2.0)

        summary = collector.summary(duration=4.0)

        assert summary.avg_data_rate == pytest.approx(0.5)

    def test_avg_data_rate_is_zero_when_nothing_is_processed(self) -> None:
        collector = MetricsCollector()

        summary = collector.summary(duration=10.0)

        assert summary.avg_data_rate == 0.0

    def test_avg_data_rate_is_zero_when_duration_is_not_positive(self) -> None:
        collector = MetricsCollector()
        _record_inference(collector, sample_id=0, arrival_time=0.0, end_time=1.0)

        summary = collector.summary(duration=0.0)

        assert summary.avg_data_rate == 0.0

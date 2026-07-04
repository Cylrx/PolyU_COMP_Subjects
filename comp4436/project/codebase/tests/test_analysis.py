from __future__ import annotations

from pathlib import Path

from config import PlotConfig
from evaluation.analysis import (
    _TradeoffPoint,
    _pareto_frontier_points,
    _prepare_admission_tradeoff_points,
    generate_admission_tradeoff_plot,
)
from evaluation.metrics import MetricsSummary


def _summary(*, accuracy: float, avg_latency: float) -> MetricsSummary:
    return MetricsSummary(
        total_arrivals=0,
        total_processed=0,
        total_dropped=0,
        drop_rate=0.0,
        avg_data_rate=0.0,
        avg_latency=avg_latency,
        p50_latency=avg_latency,
        p95_latency=avg_latency,
        p99_latency=avg_latency,
        avg_aoi=avg_latency,
        accuracy=accuracy,
        deadline_miss_rate=0.0,
        edge_utilization=0.0,
        cloud_utilization=0.0,
        offload_ratio=0.0,
        per_model_accuracy={},
        per_model_avg_latency={},
        per_model_count={},
        drops_by_reason={},
        model_switch_count=0,
        total_model_load_time=0.0,
        queue_length_over_time=[],
        data_rate_over_time=[],
        accuracy_over_time=[],
        deadline_misses_over_time=[],
        all_latencies=[],
    )


def test_prepare_admission_tradeoff_points_uses_combined_abbreviations() -> None:
    results = {
        "DropOld | FixedEdge": _summary(accuracy=0.90, avg_latency=0.200),
        "DropTail | MultiStepOptimal": _summary(accuracy=0.88, avg_latency=0.150),
    }

    points = _prepare_admission_tradeoff_points(results)

    assert [
        (point.label, point.dispatcher_name, point.accuracy_pct, point.latency_ms)
        for point in points
    ] == [
        ("DrO+FxE", "FixedEdge", 90.0, 200.0),
        ("DrT+MSO", "MultiStepOptimal", 88.0, 150.0),
    ]


def test_pareto_frontier_points_excludes_dominated_points_and_sorts_by_latency() -> None:
    points = [
        _TradeoffPoint("DrO+AdQ", "AdaptiveQueue", accuracy_pct=88.0, latency_ms=140.0),
        _TradeoffPoint("DrT+FxO", "FixedOffload", accuracy_pct=85.0, latency_ms=100.0),
        _TradeoffPoint("DrO+1SO", "OneStepOptimal", accuracy_pct=90.0, latency_ms=120.0),
        _TradeoffPoint("DrT+MSO", "MultiStepOptimal", accuracy_pct=92.0, latency_ms=150.0),
        _TradeoffPoint("DrO+FxE", "FixedEdge", accuracy_pct=90.0, latency_ms=130.0),
    ]

    frontier = _pareto_frontier_points(points)

    assert [point.label for point in frontier] == [
        "DrT+FxO",
        "DrO+1SO",
        "DrT+MSO",
    ]


def test_generate_admission_tradeoff_plot_writes_png(tmp_path: Path) -> None:
    results = {
        "DropOld | FixedEdge": _summary(accuracy=0.91, avg_latency=0.220),
        "DropOld | OneStepOptimal": _summary(accuracy=0.89, avg_latency=0.180),
        "DropTail | MultiStepOptimal": _summary(accuracy=0.90, avg_latency=0.160),
        "DropTail | AdaptiveQueue": _summary(accuracy=0.82, avg_latency=0.240),
    }
    config = PlotConfig(
        styles=("science", "no-latex", "nature", "discrete-rainbow-12"),
        output_format="png",
        admission_tradeoff_figure_width=4.0,
        admission_tradeoff_figure_height=4.8,
    )

    generate_admission_tradeoff_plot(results, tmp_path, config=config)

    assert (tmp_path / "admission_tradeoff.png").exists()

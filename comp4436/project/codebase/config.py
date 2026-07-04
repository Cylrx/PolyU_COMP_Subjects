from __future__ import annotations

from dataclasses import dataclass

# Torch.manual_seed(3407) is all you need XD
# https://arxiv.org/abs/2109.08203
_SEED = 3407

@dataclass(frozen=True)
class ProfilingConfig:
    # Multiplies profiled inference latencies before they are stored in
    # generated profile caches or shown in profiling summaries.
    edge_latency_scale_factor: float = 1.0
    cloud_latency_scale_factor: float = 1.0


@dataclass(frozen=True)
class ModelVariantBlacklistConfig:
    # Remove model variants before experiments are built.
    # Downstream components only see the surviving variants.
    enabled: bool = False
    disabled_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimulationConfig:
    # System-level parameters for a single simulation run.
    # Arrival rate is not configured here; each ArrivalPattern owns it.
    duration: float = 20.0  # Simulated seconds to run.
    deadline_budget: float = 0.35  # Seconds after arrival before a sample is stale.
    queue_capacity: int = 50  # Maximum items in the processing queue.
    edge_model_load_time: float = 0.25  # Seconds to switch models on edge.
    cloud_capacity: int = 1  # Maximum concurrent cloud inferences.
    network_base_rtt: float = 0.1  # Base edge<->cloud round-trip time in seconds.
    network_jitter_std: float = 0.02  # Stddev of Gaussian network jitter.
    seed: int = _SEED  # Random seed for reproducibility.
    n_repetitions: int = 5  # Run each experiment N times and average results.


@dataclass(frozen=True)
class ArrivalConfig:
    # Shared arrival-rate parameters for all patterns.
    rate: float = 80.0  # Samples per second (used by all patterns).
    seed: int = _SEED  # RNG seed for stochastic patterns.

    # GammaArrival-specific.
    gamma_shape: float = 0.05  # Shape parameter (lower = burstier).

    # BurstyArrival-specific.
    bursty_high_rate: float = 160.0  # Burst-phase rate.
    bursty_low_rate: float = 20.0  # Calm-phase rate.
    bursty_period: float = 5.0  # Cycle length in seconds.
    bursty_burst_ratio: float = 0.4  # Fraction of cycle in burst phase.


@dataclass(frozen=True)
class OneStepOptimalConfig:
    # One-step optimal dispatch tuning parameters.
    # Lower value_scale_multiplier or higher backlog_weight favors faster models.
    value_scale_multiplier: float = 1.0
    backlog_weight: float = 1.0


@dataclass(frozen=True)
class MultiStepOptimalConfig:
    # Multi-step optimal dispatch tuning parameters.
    max_backlog_window: int = 50  # Samples included in planning window.
    time_quantum: float = 0.03  # Time discretization in seconds.
    service_time_penalty: float = 1.0  # Lambda in accuracy - lambda * service_time.


@dataclass(frozen=True)
class AdmissionExperimentConfig:
    # Admission control ablation: single arrival pattern, all dispatchers.
    arrival_pattern: str = "Gamma"  # One of: Uniform, Poisson, Gamma, Fixed, Bursty.

    # Metric keys shown in console and Typst tables (max 3 recommended).
    # Available: data_rate, accuracy, avg_latency, avg_aoi,
    #            p95_latency, p99_latency, deadline_miss_rate.
    table_metrics: tuple[str, ...] = ("data_rate", "accuracy", "avg_latency")


@dataclass(frozen=True)
class PlotConfig:
    # Time-series binning interval in simulated seconds.
    metric_interval: float = 1.0

    # Visual parameters (uniform across entire figure).
    font_size: float = 8.5
    figure_width: float = 7.0
    dispatch_figure_height: float = 3.0
    queue_timeline_figure_height: float = 1.0
    admission_tradeoff_figure_width: float = 3.6
    admission_tradeoff_figure_height: float = 236
    styles: tuple[str, ...] = ("science", "nature", "discrete-rainbow-12", "no-latex")
    reverse_colors: bool = False  # Reverse the color cycle ordering.
    markers: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
    admission_tradeoff_non_frontier_marker: str = "o"
    admission_tradeoff_frontier_markers: tuple[str, ...] = ("s", "^", "D", "v", "P", "X")
    admission_tradeoff_non_frontier_marker_size: float = 36.0
    admission_tradeoff_frontier_marker_size: float = 42.0
    admission_tradeoff_marker_edgewidth: float = 0.7
    admission_tradeoff_frontier_linewidth: float = 1.0
    admission_tradeoff_xlabel: str = "Latency (ms)"
    admission_tradeoff_ylabel: str = r"Accuracy (\%)"

    # Output settings.
    output_format: str = "both"  # "pdf", "png", or "both"
    dpi: int = 300

    # Algorithm renaming.  Keys present in the map are included and
    # renamed to the corresponding value; keys absent are excluded.
    label_map: tuple[tuple[str, str], ...] = (
        ("DropOld + AdaptiveQueue", "AdaptQueue"),
        ("DropOld + AdaptiveDeadline", "AdaptDeadline"),
        ("DropOld + OneStepOptimal", "OneStepOpt"),
        ("DropOld + MultiStepOptimal", "MultiStepOpt"),
    )

    # Short bar-chart x-tick labels (same order as label_map).
    bar_labels: tuple[str, ...] = ("AdQ", "AdD", "1SO", "MSO")

    # Subplot spacing.
    subplot_hspace: float = 0.15  # Vertical gap between rows (incl. legend).
    subplot_wspace: float = 0.25  # Horizontal gap between columns.

    # Axis and bar outline weight (shared).
    axis_linewidth: float = 0.8

    # Bar hatch patterns (one per algorithm, cycled).
    bar_hatches: tuple[str, ...] = ("//", "\\\\", "xx", "++", "||", "--")

    # Grid line color for line charts (bar charts have no grid).
    grid_color: str = "#cccccc"

    # Log-scale toggles per subplot (data_rate_line, data_rate_bar,
    # accuracy_line, accuracy_bar, deadline_line, deadline_bar).
    log_scale: tuple[bool, ...] = (False, False, False, False, False, True)

    # Y-axis limits per subplot, same order as log_scale.
    # Each entry is (lower_bound, upper_bound); None means auto.
    ylim: tuple[tuple[float | None, float | None], ...] = (
        (None, None), (None, None), (None, None),
        (0.7, None), (None, None), (None, 150),
    )

    # Queue timeline settings.
    queue_timeline_pattern: str = "Gamma"  # Arrival pattern shown in the figure.
    queue_ema_span: int = 200  # EMA span (higher = smoother).

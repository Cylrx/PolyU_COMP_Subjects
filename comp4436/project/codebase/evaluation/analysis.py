"""Visualization module for dispatch experiment results.

Generates a single combined figure per invocation with 6 subplots
(3 rows x 2 columns) plus a shared legend strip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402, F401  -- registers styles with matplotlib

from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from config import PlotConfig
    from evaluation.metrics import MetricsSummary


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_RESULT_KEY_SEP = " | "


def _apply_styles(config: PlotConfig) -> None:
    """Load SciencePlots style stack then enforce uniform font sizes."""
    plt.style.use(list(config.styles))
    size = config.font_size
    plt.rcParams.update({
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "axes.linewidth": config.axis_linewidth,
    })


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    config: PlotConfig,
) -> None:
    if config.output_format in ("pdf", "both"):
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    if config.output_format in ("png", "both"):
        fig.savefig(output_dir / f"{name}.png", dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def _format_log_tick_plain(value: float, _: object) -> str:
    """Format log-scale ticks as plain numbers instead of exponential form."""
    if value <= 0:
        return ""
    if abs(value - round(value)) < 1e-12:
        return f"{int(round(value))}"
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Result preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TradeoffPoint:
    """One admission/dispatcher point in the accuracy-latency plane."""

    label: str
    dispatcher_name: str
    accuracy_pct: float
    latency_ms: float


def _prepare_results(
    results: dict[str, MetricsSummary],
    config: PlotConfig,
) -> dict[str, MetricsSummary]:
    """Strip label prefix, apply exclusion filter, return display-name dict."""
    mapping = dict(config.label_map)
    prepared: dict[str, MetricsSummary] = {}
    for key, summary in results.items():
        if key in mapping:
            prepared[mapping[key]] = summary
    return prepared


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------


def _add_shared_legend(
    ax: Axes,
    handles: list[object],
    labels: list[str],
) -> None:
    """Render a centered legend strip if there is anything to show."""
    if not handles:
        return
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=min(len(handles), 6),
        frameon=True,
        edgecolor="lightgray",
        fancybox=True,
        borderpad=0.4,
        columnspacing=1.2,
    )
    legend.get_frame().set_linewidth(0.5)


def _add_admission_tradeoff_legends(
    plot_ax: Axes,
    dispatcher_colors: dict[str, str],
    frontier_points: list[_TradeoffPoint],
    frontier_markers: tuple[str, ...],
    config: PlotConfig,
) -> None:
    """Render separate legends for dispatcher color and frontier shape."""
    from experiments._dispatchers import SPECS as dispatcher_specs

    color_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker=config.admission_tradeoff_non_frontier_marker,
            markersize=6,
            markerfacecolor=dispatcher_colors[spec.name],
            markeredgecolor="black",
            markeredgewidth=config.admission_tradeoff_marker_edgewidth,
            label=spec.abbreviation,
        )
        for spec in dispatcher_specs
    ]
    compact_legend_kwargs = {
        "frameon": True,
        "edgecolor": "lightgray",
        "fancybox": True,
        "borderpad": 0.35,
        "labelspacing": 0.25,
        "columnspacing": 0.7,
        "handlelength": 1.1,
        "handletextpad": 0.3,
        "borderaxespad": 0.1,
    }
    color_legend = plot_ax.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.30, -0.16),
        ncol=3,
        title="Color = Dispatcher",
        **compact_legend_kwargs,
    )
    color_legend.get_frame().set_linewidth(0.5)
    plot_ax.add_artist(color_legend)

    if not frontier_points:
        return

    shape_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker=frontier_markers[idx % len(frontier_markers)],
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=config.admission_tradeoff_marker_edgewidth,
            label=point.label,
        )
        for idx, point in enumerate(frontier_points)
    ]
    shape_legend = plot_ax.legend(
        handles=shape_handles,
        loc="upper center",
        bbox_to_anchor=(0.77, -0.16),
        ncol=1,
        title="Pareto Frontier",
        **compact_legend_kwargs,
    )
    shape_legend.get_frame().set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Subplot helpers
# ---------------------------------------------------------------------------


def _get_colors(config: PlotConfig) -> list[str]:
    """Return the color cycle, reversed if config requests it."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return list(reversed(colors)) if config.reverse_colors else colors


def _dispatcher_color_map(config: PlotConfig) -> dict[str, str]:
    """Assign a stable color to each dispatcher across admission policies."""
    from experiments._dispatchers import SPECS as dispatcher_specs

    colors = _get_colors(config)
    return {
        spec.name: colors[idx % len(colors)]
        for idx, spec in enumerate(dispatcher_specs)
    }


def _plot_line_chart(
    ax: Axes,
    results: dict[str, MetricsSummary],
    series_getter: Callable[[MetricsSummary], list[tuple[float, float | int]]],
    ylabel: str,
    config: PlotConfig,
) -> list[Line2D]:
    """Plot one metric's time series for all algorithms on *ax*.

    Returns the ``Line2D`` handles needed for the shared legend.
    """
    colors = _get_colors(config)
    lines: list[Line2D] = []
    for idx, (name, summary) in enumerate(results.items()):
        series = series_getter(summary)
        if not series:
            continue
        times, values = zip(*series)
        (line,) = ax.plot(
            times,
            values,
            label=name,
            color=colors[idx % len(colors)],
            marker=config.markers[idx % len(config.markers)],
            markersize=3,
        )
        lines.append(line)
    ax.set_ylabel(ylabel, rotation=90)
    ax.grid(True, color=config.grid_color, linewidth=0.5)
    ax.set_axisbelow(True)
    return lines


def _plot_bar_chart(
    ax: Axes,
    results: dict[str, MetricsSummary],
    metric_getter: Callable[[MetricsSummary], float],
    ylabel: str,
    config: PlotConfig,
) -> None:
    """Plot one aggregate metric as a bar chart on *ax*."""
    colors = _get_colors(config)
    names = list(results.keys())
    values = [metric_getter(s) for s in results.values()]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    hatches = [config.bar_hatches[i % len(config.bar_hatches)] for i in range(len(names))]

    x = range(len(names))
    bars = ax.bar(
        x, values, color=bar_colors,
        edgecolor="black", linewidth=config.axis_linewidth,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_ylabel(ylabel, rotation=90)
    ax.set_xticks(list(x))
    ax.set_xticklabels([])
    ax.grid(False)


# ---------------------------------------------------------------------------
# Queue timeline helpers
# ---------------------------------------------------------------------------


def _ema(values: list[float], span: int) -> list[float]:
    """Exponential moving average with the given *span*."""
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


def _deadline_miss_series(summary: MetricsSummary) -> list[tuple[float, float]]:
    """Return deadline-miss time series as floats for plotting."""
    return [(t, float(v)) for t, v in summary.deadline_misses_over_time]


def _deadline_miss_percent(summary: MetricsSummary) -> float:
    """Return aggregate deadline-miss ratio in percentage points."""
    return summary.deadline_miss_rate * 100


# ---------------------------------------------------------------------------
# Admission tradeoff helpers
# ---------------------------------------------------------------------------


def _prepare_admission_tradeoff_points(
    results: dict[str, MetricsSummary],
) -> list[_TradeoffPoint]:
    """Return admission tradeoff points in canonical suite order."""
    from experiments._admissions import SPECS as admission_specs
    from experiments._dispatchers import SPECS as dispatcher_specs

    points: list[_TradeoffPoint] = []
    for admission_spec in admission_specs:
        for dispatcher_spec in dispatcher_specs:
            key = f"{admission_spec.name}{_RESULT_KEY_SEP}{dispatcher_spec.name}"
            summary = results.get(key)
            if summary is None:
                continue
            points.append(
                _TradeoffPoint(
                    label=f"{admission_spec.abbreviation}+{dispatcher_spec.abbreviation}",
                    dispatcher_name=dispatcher_spec.name,
                    accuracy_pct=summary.accuracy * 100.0,
                    latency_ms=summary.avg_latency * 1000.0,
                )
            )
    return points


def _dominates(lhs: _TradeoffPoint, rhs: _TradeoffPoint) -> bool:
    """Return whether *lhs* dominates *rhs* on accuracy up / latency down."""
    return (
        lhs.accuracy_pct >= rhs.accuracy_pct
        and lhs.latency_ms <= rhs.latency_ms
        and (
            lhs.accuracy_pct > rhs.accuracy_pct
            or lhs.latency_ms < rhs.latency_ms
        )
    )


def _pareto_frontier_points(points: list[_TradeoffPoint]) -> list[_TradeoffPoint]:
    """Return the non-dominated tradeoff points sorted by latency."""
    frontier = [
        candidate
        for candidate in points
        if not any(_dominates(other, candidate) for other in points if other is not candidate)
    ]
    return sorted(frontier, key=lambda point: (point.latency_ms, point.accuracy_pct))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_all_plots(
    results: dict[str, MetricsSummary],
    output_dir: Path,
    config: PlotConfig | None = None,
) -> None:
    """Generate the combined dispatch comparison figure.

    One figure is produced per call.  ``dispatch.report()`` calls this once
    per arrival pattern, yielding one figure per pattern.
    """
    if config is None:
        from config import PlotConfig as _PlotConfig

        config = _PlotConfig()

    _apply_styles(config)
    prepared = _prepare_results(results, config)
    if not prepared:
        return

    fig = plt.figure(figsize=(config.figure_width, config.dispatch_figure_height))
    gs = GridSpec(
        nrows=4,
        ncols=2,
        figure=fig,
        width_ratios=[2.2, 1.0],
        height_ratios=[0.15, 1.0, 1.0, 1.0],
        hspace=config.subplot_hspace,
        wspace=config.subplot_wspace,
    )

    # -- Legend strip (row 0, spans both columns) ----------------------------
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.set_axis_off()

    # -- Data subplots -------------------------------------------------------
    line_axes = [fig.add_subplot(gs[row, 0]) for row in range(1, 4)]
    bar_axes = [fig.add_subplot(gs[row, 1]) for row in range(1, 4)]

    for ax in line_axes[1:]:
        ax.sharex(line_axes[0])

    row_specs = (
        (lambda s: s.data_rate_over_time, "DR", lambda s: s.avg_data_rate, "Avg. DR"),
        (lambda s: s.accuracy_over_time, "Acc.", lambda s: s.accuracy, "Avg. Acc."),
        (_deadline_miss_series, "Miss", _deadline_miss_percent, "\\%Miss"),
    )

    lines: list[Line2D] = []
    for row_idx, (line_ax, bar_ax, spec) in enumerate(zip(line_axes, bar_axes, row_specs)):
        line_series_getter, line_ylabel, bar_metric_getter, bar_ylabel = spec
        row_lines = _plot_line_chart(
            line_ax,
            prepared,
            line_series_getter,
            line_ylabel,
            config,
        )
        _plot_bar_chart(bar_ax, prepared, bar_metric_getter, bar_ylabel, config)
        if row_idx == 0:
            lines = row_lines
        if row_idx < len(row_specs) - 1:
            line_ax.tick_params(labelbottom=False)

    line_axes[-1].set_xlabel("Time (sec)")
    bar_axes[-1].set_xticks(list(range(len(prepared))))
    bar_axes[-1].set_xticklabels(list(config.bar_labels[: len(prepared)]))
    bar_axes[-1].set_xlabel("Algorithms")

    # -- Per-subplot axis tweaks (log scale, ylim) ----------------------------
    ordered_axes = [
        line_axes[0], bar_axes[0],
        line_axes[1], bar_axes[1],
        line_axes[2], bar_axes[2],
    ]
    for i, ax in enumerate(ordered_axes):
        if i < len(config.log_scale) and config.log_scale[i]:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick_plain))
            ax.yaxis.set_minor_formatter(NullFormatter())
        if i < len(config.ylim):
            lo, hi = config.ylim[i]
            if lo is not None or hi is not None:
                ax.set_ylim(bottom=lo, top=hi)

    # -- Legend strip ---------------------------------------------------------
    _add_shared_legend(ax_legend, lines, list(prepared.keys()))

    _save_figure(fig, output_dir, "dispatch_comparison", config)


def generate_queue_timeline(
    first_run: dict[str, MetricsSummary],
    pattern_name: str,
    output_dir: Path,
    config: PlotConfig | None = None,
) -> None:
    """Generate queue length timeline for a single arrival pattern.

    For each algorithm, a translucent raw line from the first repetition is
    drawn as background context, with a prominent EMA curve on top.

    ``first_run`` maps raw algorithm keys (matching ``PlotConfig.label_map``
    keys) to the first repetition's summary.
    """
    if config is None:
        from config import PlotConfig as _PlotConfig

        config = _PlotConfig()

    _apply_styles(config)
    colors = _get_colors(config)

    fig = plt.figure(figsize=(config.figure_width, config.queue_timeline_figure_height))
    gs = GridSpec(
        nrows=2,
        ncols=1,
        figure=fig,
        height_ratios=[0.30, 1.0],
        hspace=config.subplot_hspace,
    )

    ax_legend = fig.add_subplot(gs[0])
    ax_legend.set_axis_off()

    ax = fig.add_subplot(gs[1])
    lines_for_legend: list[Line2D] = []

    for algo_idx, (raw_key, display_name) in enumerate(config.label_map):
        if raw_key not in first_run:
            continue
        pts = first_run[raw_key].queue_length_over_time
        if not pts:
            continue

        color = colors[algo_idx % len(colors)]
        times, sizes = zip(*pts)
        sizes_f = [float(s) for s in sizes]

        # Raw line: translucent background.
        ax.plot(times, sizes_f, color=color, alpha=0.5, linewidth=0.5, zorder=1)

        # EMA curve: prominent foreground.
        ema_vals = _ema(sizes_f, span=config.queue_ema_span)
        (line,) = ax.plot(
            times, ema_vals,
            color=color, linewidth=1.2, label=display_name, zorder=2,
        )
        lines_for_legend.append(line)

    ax.set_ylabel("Queue Len.", rotation=90)
    ax.grid(True, color=config.grid_color, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time (sec)")

    _add_shared_legend(
        ax_legend,
        lines_for_legend,
        [line.get_label() for line in lines_for_legend],
    )

    _save_figure(fig, output_dir, "queue_timeline", config)


def generate_admission_tradeoff_plot(
    results: dict[str, MetricsSummary],
    output_dir: Path,
    config: PlotConfig | None = None,
) -> None:
    """Generate the admission accuracy-latency tradeoff figure."""
    if config is None:
        from config import PlotConfig as _PlotConfig

        config = _PlotConfig()

    _apply_styles(config)
    points = _prepare_admission_tradeoff_points(results)
    if not points:
        return

    frontier_points = _pareto_frontier_points(points)
    frontier_labels = {point.label for point in frontier_points}
    non_frontier_points = [
        point for point in points if point.label not in frontier_labels
    ]
    dispatcher_colors = _dispatcher_color_map(config)

    fig = plt.figure(
        figsize=(
            config.admission_tradeoff_figure_width,
            config.admission_tradeoff_figure_height,
        )
    )
    gs = GridSpec(
        nrows=1,
        ncols=1,
        figure=fig,
    )
    ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(1)

    if non_frontier_points:
        ax.scatter(
            [point.latency_ms for point in non_frontier_points],
            [point.accuracy_pct for point in non_frontier_points],
            c=[
                dispatcher_colors[point.dispatcher_name]
                for point in non_frontier_points
            ],
            marker=config.admission_tradeoff_non_frontier_marker,
            edgecolors="black",
            linewidths=config.admission_tradeoff_marker_edgewidth,
            s=config.admission_tradeoff_non_frontier_marker_size,
            zorder=2,
        )

    for idx, point in enumerate(frontier_points):
        ax.scatter(
            [point.latency_ms],
            [point.accuracy_pct],
            color=dispatcher_colors[point.dispatcher_name],
            marker=(
                config.admission_tradeoff_frontier_markers[
                    idx % len(config.admission_tradeoff_frontier_markers)
                ]
            ),
            edgecolors="black",
            linewidths=config.admission_tradeoff_marker_edgewidth,
            s=config.admission_tradeoff_frontier_marker_size,
            zorder=4,
        )

    if len(frontier_points) >= 2:
        ax.plot(
            [point.latency_ms for point in frontier_points],
            [point.accuracy_pct for point in frontier_points],
            color="black",
            linewidth=config.admission_tradeoff_frontier_linewidth,
            zorder=3,
        )

    ax.set_xlabel(config.admission_tradeoff_xlabel)
    ax.set_ylabel(config.admission_tradeoff_ylabel, rotation=90)
    ax.grid(True, color=config.grid_color, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.08)

    _add_admission_tradeoff_legends(
        ax,
        dispatcher_colors,
        frontier_points,
        config.admission_tradeoff_frontier_markers,
        config,
    )

    _save_figure(fig, output_dir, "admission_tradeoff", config)

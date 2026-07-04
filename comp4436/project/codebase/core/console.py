"""Shared rich Console and formatting utilities.

All console output in the project goes through this module.
No raw print() calls elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable

    from core.compat import EnvironmentStatus
    from core.types import ModelVariant, ProfileMetadata
    from evaluation.metrics import MetricsSummary

console = Console()


# -- Status messages ----------------------------------------------------------


def status(msg: str) -> None:
    console.print(f"  [blue]▶[/] {msg}")


def success(msg: str) -> None:
    console.print(f"  [green]✔[/] {msg}")


def warning(msg: str) -> None:
    console.print(f"  [yellow]⚠[/] {msg}")


def error(msg: str) -> None:
    console.print(f"  [red]✗[/] {msg}")


class SampleProgressDisplay:
    """Single-line live progress for long-running scheduling decisions."""

    def __init__(self, label: str, total_samples: int) -> None:
        self._label = label
        self._total_samples = total_samples
        self._current_sample = 0
        self._live: Live | None = None

    def __enter__(self) -> SampleProgressDisplay:
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._live is None:
            return
        if exc_type is None:
            self._current_sample = self._total_samples
            self._live.update(self._render(done=True), refresh=True)
        self._live.stop()
        self._live = None

    def update(self, current_sample: int) -> None:
        bounded = max(0, min(current_sample, self._total_samples))
        if bounded == self._current_sample or self._live is None:
            return
        self._current_sample = bounded
        self._live.update(self._render(), refresh=False)

    def _render(self, *, done: bool = False) -> Text:
        status_text = "done" if done else "processing"
        text = Text()
        text.append(f"{self._label}: ", style="bold cyan")
        text.append(
            f"{status_text} sample {self._current_sample}/{self._total_samples}",
            style="dim",
        )
        return text


# -- Environment display ------------------------------------------------------


def print_environment_status(env: EnvironmentStatus) -> None:
    console.print()
    console.rule("[bold]Environment Check")

    _check_line(True, f"Python {env.python_version}")
    _check_line(env.torch_available, f"PyTorch {env.torch_version or 'not installed'}")
    _check_line(env.torchvision_available, "torchvision")

    if env.torch_available:
        if env.cuda_available:
            _check_line(True, f"CUDA {env.cuda_version}")
        else:
            warning("CUDA not available (profiling on CPU)")

    console.print(f"  [dim]Device: {env.device}[/]")
    console.print()


def _check_line(ok: bool, msg: str) -> None:
    if ok:
        success(msg)
    else:
        error(msg)


# -- Profile info -------------------------------------------------------------


def print_profile_info(metadata: ProfileMetadata) -> None:
    info = (
        f"[bold]{metadata.description}[/]\n"
        f"Kind: {metadata.kind} | "
        f"Samples: {metadata.n_samples} | "
        f"Variants: {len(metadata.model_variants)} | "
        f"Edge: {metadata.edge_device} | "
        f"Cloud: {metadata.cloud_device}\n"
        f"Created: {metadata.created_at}"
    )
    if metadata.torch_version:
        info += f" | PyTorch {metadata.torch_version}"
    console.print(Panel(info, title="Profile", border_style="blue"))


def print_model_variants(variants: list[ModelVariant]) -> None:
    """Print a summary table of model variant metrics."""
    table = _base_table(title="Model Variants")
    table.add_column("Model", style="bold", min_width=12)
    table.add_column("Accuracy", justify="right")
    table.add_column("Edge Latency", justify="right")
    table.add_column("Cloud Latency", justify="right")
    table.add_column("MACs", justify="right")

    for v in variants:
        table.add_row(
            v.name,
            f"{v.accuracy:.1%}",
            _format_ms(v.edge_avg_latency),
            _format_ms(v.cloud_avg_latency),
            _format_macs(v.macs),
        )

    console.print()
    console.print(table)


# -- Experiment results: column model ------------------------------------------


@dataclass(frozen=True)
class _Segment:
    """One rankable value within a display column."""

    extract: Callable[[MetricsSummary], float]
    format: Callable[[MetricsSummary], str]
    lower_is_better: bool | None = None


@dataclass(frozen=True)
class _Column:
    """A display column composed of independently ranked segments."""

    label: str
    segments: tuple[_Segment, ...]
    separator: str = ""
    suffix: str = ""
    justify: str = "right"


_RANK_STYLES: dict[int, str] = {1: "bold", 2: "italic"}

_RESULT_COLUMNS: tuple[_Column, ...] = (
    _Column("Proc'd", (_Segment(
        lambda s: s.total_processed, lambda s: str(s.total_processed), False),)),
    _Column("DR", (_Segment(
        lambda s: s.avg_data_rate,
        lambda s: _format_data_rate(s.avg_data_rate), False),)),
    _Column("Dropped", (_Segment(
        lambda s: s.drop_rate,
        lambda s: f"{s.total_dropped} ({s.drop_rate:.0%})", True),),
        justify="left"),
    _Column("Accuracy", (_Segment(
        lambda s: s.accuracy, lambda s: f"{s.accuracy:.1%}", False),)),
    _Column("Avg Lat", (_Segment(
        lambda s: s.avg_latency,
        lambda s: f"{s.avg_latency * 1000:.0f}ms", True),)),
    _Column("Avg AoI", (_Segment(
        lambda s: s.avg_aoi, lambda s: _format_ms(s.avg_aoi), True),)),
    _Column("P95/P99 Lat", (
        _Segment(lambda s: s.p95_latency,
                 lambda s: f"{s.p95_latency * 1000:.0f}", True),
        _Segment(lambda s: s.p99_latency,
                 lambda s: f"{s.p99_latency * 1000:.0f}", True),
    ), separator="/", suffix=" ms"),
    _Column("Miss %", (_Segment(
        lambda s: s.deadline_miss_rate,
        lambda s: f"{s.deadline_miss_rate:.0%}", True),)),
    _Column("Offload", (_Segment(
        lambda s: s.offload_ratio, lambda s: f"{s.offload_ratio:.0%}"),)),
)

# Metric-key → Column lookup for configurable table output.
# Keys match those used in AdmissionExperimentConfig.table_metrics.
COLUMN_REGISTRY: dict[str, _Column] = {
    col.label: col for col in _RESULT_COLUMNS
}
_LABEL_ALIASES: dict[str, str] = {
    "data_rate": "DR",
    "accuracy": "Accuracy",
    "avg_latency": "Avg Lat",
    "avg_aoi": "Avg AoI",
    "p95_p99_latency": "P95/P99 Lat",
    "deadline_miss_rate": "Miss %",
    "offload": "Offload",
    "processed": "Proc'd",
    "dropped": "Dropped",
}


def select_columns(keys: tuple[str, ...]) -> tuple[_Column, ...]:
    """Select result columns by metric key."""
    return tuple(COLUMN_REGISTRY[_LABEL_ALIASES[k]] for k in keys)


# -- Experiment results: ranking ----------------------------------------------


def _group_by_arrival_pattern(
    results: dict[str, MetricsSummary],
) -> list[tuple[str | None, list[tuple[str, MetricsSummary]]]]:
    """Group experiments by the arrival-pattern prefix (text before ' | ')."""
    groups: list[tuple[str | None, list[tuple[str, MetricsSummary]]]] = []
    prev_group: str | None = None
    current: list[tuple[str, MetricsSummary]] = []

    for name, summary in results.items():
        group = name.split(" | ", 1)[0] if " | " in name else None
        if current and group != prev_group:
            groups.append((prev_group, current))
            current = []
        prev_group = group
        current.append((name, summary))

    if current:
        groups.append((prev_group, current))
    return groups


def _compute_segment_ranks(
    summaries: list[MetricsSummary],
    segment: _Segment,
) -> list[int]:
    """Rank experiments for one segment: 1 = best, 2 = second-best, 0 = unranked."""
    if segment.lower_is_better is None:
        return [0] * len(summaries)

    values = [segment.extract(s) for s in summaries]
    valid = sorted({v for v in values if not math.isnan(v)})
    if not valid:
        return [0] * len(summaries)

    best = valid[0] if segment.lower_is_better else valid[-1]
    second = (valid[1] if segment.lower_is_better else valid[-2]) if len(valid) > 1 else None

    return [1 if v == best else 2 if v == second else 0 for v in values]


def _compute_group_ranks(
    summaries: list[MetricsSummary],
    columns: tuple[_Column, ...] = _RESULT_COLUMNS,
) -> list[list[list[int]]]:
    """Compute ranks indexed by [column][segment][experiment]."""
    return [
        [_compute_segment_ranks(summaries, seg) for seg in col.segments]
        for col in columns
    ]


def _render_cell(
    column: _Column,
    summary: MetricsSummary,
    segment_ranks: list[int],
) -> Text:
    """Render a column cell with rank-based Rich styling."""
    cell = Text()
    for i, segment in enumerate(column.segments):
        if i > 0:
            cell.append(column.separator)
        cell.append(segment.format(summary), style=_RANK_STYLES.get(segment_ranks[i], ""))
    if column.suffix:
        cell.append(column.suffix)
    return cell


# -- Experiment results -------------------------------------------------------


def print_experiment_results(
    results: dict[str, MetricsSummary],
    columns: tuple[_Column, ...] | None = None,
) -> None:
    """Print results with per-group best (bold) and second-best (italic) highlighting.

    When *columns* is ``None``, all default columns are shown.
    """
    active = columns if columns is not None else _RESULT_COLUMNS

    table = _base_table(title="Experiment Results")
    table.add_column("Experiment Configuration", style="bold", min_width=20)
    for col in active:
        table.add_column(col.label, justify=col.justify)

    for group_idx, (_, group_items) in enumerate(_group_by_arrival_pattern(results)):
        if group_idx > 0:
            table.add_section()

        summaries = [s for _, s in group_items]
        ranks = _compute_group_ranks(summaries, active)

        for exp_idx, (name, summary) in enumerate(group_items):
            cells: list[Text] = []
            for col_idx, col in enumerate(active):
                seg_ranks = [
                    ranks[col_idx][seg_idx][exp_idx]
                    for seg_idx in range(len(col.segments))
                ]
                cells.append(_render_cell(col, summary, seg_ranks))
            table.add_row(name, *cells)

    console.print()
    console.print(table)


def print_single_result(name: str, summary: MetricsSummary) -> None:
    table = _base_table(title=name, show_header=False)
    table.add_column("Metric", style="bold", min_width=18)
    table.add_column("Value", justify="right")

    s = summary
    table.add_row("Arrivals", str(s.total_arrivals))
    table.add_row("Processed", str(s.total_processed))
    table.add_row("Dropped", f"{s.total_dropped} ({s.drop_rate:.1%})")
    table.add_row("Accuracy", f"{s.accuracy:.2%}")
    table.add_row("Avg Latency", f"{s.avg_latency * 1000:.1f} ms")
    table.add_row("P99 Latency", f"{s.p99_latency * 1000:.1f} ms")
    table.add_row("Avg AoI", _format_ms(s.avg_aoi))
    table.add_row("Deadline Miss", f"{s.deadline_miss_rate:.1%}")
    table.add_row("Offload Ratio", f"{s.offload_ratio:.1%}")
    table.add_row("Model Switches", f"{s.model_switch_count} ({s.total_model_load_time * 1000:.1f} ms)")

    if s.drops_by_reason:
        table.add_row("Drop Reasons", ", ".join(f"{k}: {v}" for k, v in s.drops_by_reason.items()))
    if s.per_model_count:
        table.add_row("Model Usage", ", ".join(f"{k}: {v}" for k, v in s.per_model_count.items()))

    console.print()
    console.print(table)


def _base_table(
    title: str,
    *,
    show_header: bool = True,
) -> Table:
    return Table(
        title=title,
        show_header=show_header,
        show_lines=False,
        box=box.SIMPLE,
        border_style="dim",
        header_style="bold",
        pad_edge=False,
    )


def _format_ms(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    return f"{value * 1000:.1f} ms"


def _format_data_rate(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    return f"{value:.1f}"


def _format_macs(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)

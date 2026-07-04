"""Typst results table generation.

Exports experiment results as a Typst ``#figure`` block matching
the format used in the project report.  The generated file assumes
``toprule``/``midrule``/``botrule`` are already imported by the
parent document (e.g. from ``bloated-neurips``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from experiments._admissions import abbreviation_for as _admission_abbreviation_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from config import AdmissionExperimentConfig, ArrivalConfig
    from evaluation.metrics import MetricsSummary

_SEP = " | "

# Layout constants (match the report's manual formatting).
_LABEL_PAD = 20  # min width for ``[label],`` in data rows
_CONT_INDENT = 14  # continuation indent for data rows
_HEADER_CONT_INDENT = 18  # continuation indent for dispatcher-name row


# -- Data definitions ---------------------------------------------------------


@dataclass(frozen=True)
class _TypstMetric:
    """One row in the Typst results table."""

    label: str
    extract: Callable[[MetricsSummary], float]
    format: Callable[[float], str]
    lower_is_better: bool


@dataclass(frozen=True)
class _TypstDispatcher:
    """One dispatcher column in the Typst results table."""

    experiment_name: str
    abbreviation: str
    is_baseline: bool
    header_bold: bool


from experiments._dispatchers import SPECS as _DISPATCHER_SPECS

# Dispatch table: experiment names carry the "DropOld + " admission prefix.
_DISPATCHERS: tuple[_TypstDispatcher, ...] = tuple(
    _TypstDispatcher(
        f"DropOld + {s.name}", s.abbreviation, s.is_baseline, s.header_bold,
    )
    for s in _DISPATCHER_SPECS
)

_N_DISPATCHERS = len(_DISPATCHERS)

_METRICS: tuple[_TypstMetric, ...] = (
    _TypstMetric(
        "DR#sym.arrow.t",
        lambda s: s.avg_data_rate,
        lambda v: f"{v:.1f}",
        lower_is_better=False,
    ),
    _TypstMetric(
        "Acc.#sym.arrow.t",
        lambda s: s.accuracy,
        lambda v: f"{v * 100:.1f}",
        lower_is_better=False,
    ),
    _TypstMetric(
        "Lat.#sym.arrow.b",
        lambda s: s.avg_latency,
        lambda v: f"{v * 1000:.0f}",
        lower_is_better=True,
    ),
    _TypstMetric(
        "AoI#sym.arrow.b",
        lambda s: s.avg_aoi,
        lambda v: f"{v * 1000:.0f}",
        lower_is_better=True,
    ),
    _TypstMetric(
        "P95#sym.arrow.b",
        lambda s: s.p95_latency,
        lambda v: f"{v * 1000:.0f}",
        lower_is_better=True,
    ),
    _TypstMetric(
        "P99#sym.arrow.b",
        lambda s: s.p99_latency,
        lambda v: f"{v * 1000:.0f}",
        lower_is_better=True,
    ),
    _TypstMetric(
        "%Miss#sym.arrow.b",
        lambda s: s.deadline_miss_rate,
        lambda v: f"{v * 100:.1f}",
        lower_is_better=True,
    ),
)

# Config-key → metric lookup for configurable table output.
METRIC_REGISTRY: dict[str, _TypstMetric] = dict(zip(
    ("data_rate", "accuracy", "avg_latency", "avg_aoi",
     "p95_latency", "p99_latency", "deadline_miss_rate"),
    _METRICS,
))


# -- Ranking ------------------------------------------------------------------


def _rank_among_non_baselines(
    values: list[float],
    lower_is_better: bool,
) -> list[int]:
    """Rank dispatchers: 1=best, 2=second-best, 0=unranked/baseline.

    Only non-baseline dispatchers participate in ranking.
    """
    rankable = [
        (i, v)
        for i, (v, d) in enumerate(zip(values, _DISPATCHERS))
        if not d.is_baseline and not math.isnan(v)
    ]
    if not rankable:
        return [0] * len(values)

    unique = sorted({v for _, v in rankable})
    best = unique[0] if lower_is_better else unique[-1]
    second = (unique[1] if lower_is_better else unique[-2]) if len(unique) > 1 else None

    ranks = [0] * len(values)
    for i, v in rankable:
        if v == best:
            ranks[i] = 1
        elif second is not None and v == second:
            ranks[i] = 2
    return ranks


def _rank_pairwise(values: list[float], lower_is_better: bool) -> list[int]:
    """Rank exactly two values: 1 = best, 0 = not best.

    Used by the admission table where only two rows are compared.
    """
    if len(values) != 2 or any(math.isnan(v) for v in values):
        return [0] * len(values)
    if values[0] == values[1]:
        return [0, 0]
    if lower_is_better:
        best_idx = 0 if values[0] < values[1] else 1
    else:
        best_idx = 0 if values[0] > values[1] else 1
    return [1 if i == best_idx else 0 for i in range(2)]


# -- Cell formatting ----------------------------------------------------------


def _typst_cell(value: str, rank: int) -> str:
    """Wrap a cell value with Typst rank styling."""
    if rank == 1:
        return f"[*{value}*]"
    if rank == 2:
        return f"[#underline[{value}]]"
    return f"[{value}]"


# -- Section builders ---------------------------------------------------------


def _build_caption(rate: float) -> str:
    """Build the figure caption text."""
    return (
        f"Dispatch results across three arrival patterns at mean arrival rate {rate:.0f}. "
        "FxE: FixedEdge, FxO: FixedOffload (reference baselines, always use full model); "
        "AdQ: AdaptQueue, AdD: AdaptDeadline, 1S: One-Step Optimal, MS: Multi-Step Optimal. "
        "Among AdQ/AdD/1S/MS, *bold* marks best and #underline[underline] second-best. "
        "DR: throughput ($s^(-1)$); Lat., AoI, P95, P99: ms."
    )


def _pattern_label(name: str, gamma_shape: float) -> str:
    """Build the display label for a pattern header cell."""
    if name == "Gamma":
        return f"Gamma ($alpha={gamma_shape}$)"
    return name


def _build_header(pattern_labels: list[str]) -> list[str]:
    """Build the ``table.header(...)`` section."""
    n_patterns = len(pattern_labels)
    lines: list[str] = []
    lines.append("    table.header(")

    # Pattern span cells (row 0)
    first_span = (
        f"table.cell(colspan: {_N_DISPATCHERS}, align: center, "
        f"inset: (top: 5pt, bottom: 5pt))[*{pattern_labels[0]}*]"
    )
    lines.append(f"      [], {first_span},")
    for label in pattern_labels[1:]:
        span = (
            f"table.cell(colspan: {_N_DISPATCHERS}, align: center, "
            f"inset: (top: 5pt, bottom: 5pt))[*{label}*]"
        )
        lines.append(f"          {span},")

    # Hlines under each pattern group
    for i in range(n_patterns):
        start = 1 + i * _N_DISPATCHERS
        end = start + _N_DISPATCHERS
        lines.append(
            f"      table.hline(start: {start}, end: {end}, stroke: 0.05em),"
        )

    # Dispatcher abbreviation cells (row 1)
    header_cells = [
        f"[*{d.abbreviation}*]" if d.header_bold else f"[{d.abbreviation}]"
        for d in _DISPATCHERS
    ]
    cells_str = ", ".join(header_cells)
    lines.append(f"      [Metrics], {cells_str},")
    for _ in range(n_patterns - 1):
        lines.append(f"{' ' * _HEADER_CONT_INDENT}{cells_str},")

    lines.append("    ),")
    return lines


def _build_data_rows(
    results: dict[str, MetricsSummary],
    pattern_names: list[str],
) -> list[str]:
    """Build the metric data rows (one row per metric, cells across patterns)."""
    lines: list[str] = []

    for metric in _METRICS:
        pattern_cell_strs: list[str] = []

        for pattern in pattern_names:
            values: list[float] = []
            for d in _DISPATCHERS:
                key = f"{pattern}{_SEP}{d.experiment_name}"
                summary = results.get(key)
                values.append(float("nan") if summary is None else metric.extract(summary))

            ranks = _rank_among_non_baselines(values, metric.lower_is_better)
            cells = [
                "[--]" if math.isnan(v) else _typst_cell(metric.format(v), rank)
                for v, rank in zip(values, ranks)
            ]
            pattern_cell_strs.append(", ".join(cells))

        # Label + first pattern on line 1; remaining patterns on continuation lines
        label = f"[{metric.label}],"
        pad = max(1, _LABEL_PAD - len(label))
        lines.append(f"    {label}{' ' * pad}{pattern_cell_strs[0]},")
        for pcs in pattern_cell_strs[1:]:
            lines.append(f"{' ' * _CONT_INDENT}{pcs},")

    return lines


# -- Public API ---------------------------------------------------------------


def generate_typst_table(
    results: dict[str, MetricsSummary],
    arrival_cfg: ArrivalConfig | None = None,
) -> str:
    """Generate a Typst ``#figure(...)`` block for the experiment results table.

    Returns the complete figure as a string, ready for inclusion in a Typst
    document that already imports ``toprule``/``midrule``/``botrule``.
    """
    if arrival_cfg is None:
        from config import ArrivalConfig

        arrival_cfg = ArrivalConfig()

    # Discover pattern names from result keys (preserving insertion order).
    pattern_names: list[str] = []
    for name in results:
        if _SEP not in name:
            continue
        pattern = name.split(_SEP, 1)[0]
        if pattern not in pattern_names:
            pattern_names.append(pattern)

    if not pattern_names:
        return ""

    n_patterns = len(pattern_names)
    n_cols = 1 + n_patterns * _N_DISPATCHERS
    n_metrics = len(_METRICS)
    last_data_y = 1 + n_metrics  # 0-indexed row of the last data row

    pattern_labels = [_pattern_label(p, arrival_cfg.gamma_shape) for p in pattern_names]

    # Table layout arrays
    columns = ", ".join(["auto"] * n_cols)
    align = ", ".join(["left"] + ["center"] * (n_cols - 1))

    gutter: list[str] = ["1pt"]
    for i in range(n_patterns):
        gutter.extend(["-3pt"] * (_N_DISPATCHERS - 1))
        if i < n_patterns - 1:
            gutter.append("6pt")
    col_gutter = ", ".join(gutter)

    row_gutter = ", ".join(["0pt"] * (n_metrics + 1))

    caption = _build_caption(arrival_cfg.rate)
    header = _build_header(pattern_labels)
    data = _build_data_rows(results, pattern_names)

    lines = [
        "#figure(",
        f"  caption: [{caption}],",
        "  {set text(size: 9.5pt)",
        "  show table.cell.where(y: 1): it => {",
        "    set text(size: 0.85em)",
        "    pad(top: 3pt, bottom: 3pt)[#it]",
        "  }",
        "  show table.cell.where(y: 2): it => pad(top: 2pt)[#it]",
        f"  show table.cell.where(y: {last_data_y}): it => pad(bottom: 2pt)[#it]",
        "  table(",
        f"    columns: ({columns}),",
        f"    align: ({align}),",
        "    stroke: none,",
        "    inset: (x: 2.5pt, y: 2.5pt),",
        f"    column-gutter: ({col_gutter}),",
        f"    row-gutter: ({row_gutter}),",
        "    toprule,",
        *header,
        "    midrule,",
        *data,
        "    botrule,",
        "  )},",
        ") <tab:results>",
    ]
    return "\n".join(lines) + "\n"


# -- Admission table ----------------------------------------------------------

# Dispatchers for the admission table (plain names, no admission prefix).
_ADMISSION_DISPATCHERS: tuple[_TypstDispatcher, ...] = tuple(
    _TypstDispatcher(s.name, s.abbreviation, s.is_baseline, s.header_bold)
    for s in _DISPATCHER_SPECS
)


def _build_admission_caption(
    metrics: list[_TypstMetric],
    arrival_pattern: str,
    rate: float,
) -> str:
    return (
        f"Admission control ablation under {arrival_pattern} arrival "
        f"(rate\u2009=\u2009{rate:.0f}). "
        "FxE: FixedEdge, FxO: FixedOffload; "
        "AdQ: AdaptQueue, AdD: AdaptDeadline, 1SO: One-Step Optimal, "
        "MSO: Multi-Step Optimal. "
        "*Bold* marks the better admission policy per dispatcher. "
        "DR: throughput ($s^(-1)$); Acc.: %; Lat.: ms."
    )


def _build_admission_header(
    metric_labels: list[str],
    dispatchers: tuple[_TypstDispatcher, ...],
) -> list[str]:
    """Build header for the admission table (metric groups × dispatchers)."""
    n_disp = len(dispatchers)
    n_metrics = len(metric_labels)
    lines: list[str] = []
    lines.append("    table.header(")

    # Row 0: metric group spans
    first_span = (
        f"table.cell(colspan: {n_disp}, align: center, "
        f"inset: (top: 5pt, bottom: 5pt))[*{metric_labels[0]}*]"
    )
    lines.append(f"      [], {first_span},")
    for label in metric_labels[1:]:
        span = (
            f"table.cell(colspan: {n_disp}, align: center, "
            f"inset: (top: 5pt, bottom: 5pt))[*{label}*]"
        )
        lines.append(f"          {span},")

    # Hlines under each metric group
    for i in range(n_metrics):
        start = 1 + i * n_disp
        end = start + n_disp
        lines.append(
            f"      table.hline(start: {start}, end: {end}, stroke: 0.05em),"
        )

    # Row 1: dispatcher abbreviation cells (repeated per metric)
    header_cells = [
        f"[*{d.abbreviation}*]" if d.header_bold else f"[{d.abbreviation}]"
        for d in dispatchers
    ]
    cells_str = ", ".join(header_cells)
    lines.append(f"      [Policy], {cells_str},")
    for _ in range(n_metrics - 1):
        lines.append(f"{' ' * _HEADER_CONT_INDENT}{cells_str},")

    lines.append("    ),")
    return lines


def _build_admission_data_rows(
    results: dict[str, MetricsSummary],
    admission_names: list[str],
    metrics: list[_TypstMetric],
    dispatchers: tuple[_TypstDispatcher, ...],
) -> list[str]:
    """Build data rows for the admission table (one row per admission policy)."""
    lines: list[str] = []

    for adm_name in admission_names:
        metric_cell_strs: list[str] = []

        for metric in metrics:
            cells: list[str] = []
            for d in dispatchers:
                # Collect values across all admissions for this (dispatcher, metric)
                col_values = [
                    _extract_value(results, a, d.experiment_name, metric)
                    for a in admission_names
                ]
                adm_idx = admission_names.index(adm_name)
                ranks = _rank_pairwise(col_values, metric.lower_is_better)
                v = col_values[adm_idx]
                cells.append(
                    "[--]" if math.isnan(v)
                    else _typst_cell(metric.format(v), ranks[adm_idx])
                )
            metric_cell_strs.append(", ".join(cells))

        abbr = _admission_abbreviation_for(adm_name)
        label = f"[{abbr}],"
        pad = max(1, _LABEL_PAD - len(label))
        lines.append(f"    {label}{' ' * pad}{metric_cell_strs[0]},")
        for mcs in metric_cell_strs[1:]:
            lines.append(f"{' ' * _CONT_INDENT}{mcs},")

    return lines


def _extract_value(
    results: dict[str, MetricsSummary],
    admission_name: str,
    dispatcher_name: str,
    metric: _TypstMetric,
) -> float:
    key = f"{admission_name}{_SEP}{dispatcher_name}"
    summary = results.get(key)
    return float("nan") if summary is None else metric.extract(summary)


def generate_admission_typst_table(
    results: dict[str, MetricsSummary],
    adm_cfg: AdmissionExperimentConfig | None = None,
) -> str:
    """Generate a Typst ``#figure(...)`` block for the admission ablation table.

    Column groups are *metrics*, rows are *admission policies*, and each
    group contains one column per dispatcher.
    """
    if adm_cfg is None:
        from config import AdmissionExperimentConfig

        adm_cfg = AdmissionExperimentConfig()

    # Resolve selected metrics from config keys.
    metrics = [METRIC_REGISTRY[k] for k in adm_cfg.table_metrics]
    dispatchers = _ADMISSION_DISPATCHERS
    n_disp = len(dispatchers)
    n_metrics = len(metrics)

    # Discover admission names from result keys (preserving insertion order).
    admission_names: list[str] = []
    for name in results:
        if _SEP not in name:
            continue
        adm = name.split(_SEP, 1)[0]
        if adm not in admission_names:
            admission_names.append(adm)

    if not admission_names:
        return ""

    n_admissions = len(admission_names)
    n_cols = 1 + n_metrics * n_disp
    last_data_y = 1 + n_admissions  # 0-indexed

    # Table layout arrays
    columns = ", ".join(["auto"] * n_cols)
    align_parts = ", ".join(["left"] + ["center"] * (n_cols - 1))

    gutter: list[str] = ["1pt"]
    for i in range(n_metrics):
        gutter.extend(["-3pt"] * (n_disp - 1))
        if i < n_metrics - 1:
            gutter.append("6pt")
    col_gutter = ", ".join(gutter)

    row_gutter = ", ".join(["0pt"] * (n_admissions + 1))

    metric_labels = [m.label for m in metrics]

    from config import ArrivalConfig

    arrival_cfg = ArrivalConfig()
    caption = _build_admission_caption(metrics, adm_cfg.arrival_pattern, arrival_cfg.rate)
    header = _build_admission_header(metric_labels, dispatchers)
    data = _build_admission_data_rows(results, admission_names, metrics, dispatchers)

    lines = [
        "#figure(",
        f"  caption: [{caption}],",
        "  {set text(size: 9.5pt)",
        "  show table.cell.where(y: 1): it => {",
        "    set text(size: 0.85em)",
        "    pad(top: 3pt, bottom: 3pt)[#it]",
        "  }",
        "  show table.cell.where(y: 2): it => pad(top: 2pt)[#it]",
        f"  show table.cell.where(y: {last_data_y}): it => pad(bottom: 2pt)[#it]",
        "  table(",
        f"    columns: ({columns}),",
        f"    align: ({align_parts}),",
        "    stroke: none,",
        "    inset: (x: 2.5pt, y: 2.5pt),",
        f"    column-gutter: ({col_gutter}),",
        f"    row-gutter: ({row_gutter}),",
        "    toprule,",
        *header,
        "    midrule,",
        *data,
        "    botrule,",
        "  )},",
        ") <tab:admission>",
    ]
    return "\n".join(lines) + "\n"

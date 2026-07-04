"""Admission control ablation suite.

Tests DropOld vs DropTail across all 6 dispatchers under a single
arrival pattern.  The two axes are admission × dispatcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config import AdmissionExperimentConfig, ArrivalConfig, PlotConfig, SimulationConfig
from core.console import select_columns
from evaluation.analysis import generate_admission_tradeoff_plot
from evaluation.metrics import MetricsSummary
from evaluation.typst_table import generate_admission_typst_table
from experiments._admissions import build_admissions
from experiments._dispatchers import build_dispatchers
from experiments.runner import ExperimentConfig
from strategies.arrival import (
    BurstyArrival,
    FixedIntervalArrival,
    GammaArrival,
    PoissonArrival,
    UniformArrival,
)

if TYPE_CHECKING:
    from core.console import _Column
    from core.types import ProfileCache
    from strategies.protocols import ArrivalPattern

_SEP = " | "

def _make_arrival_pattern(name: str, cfg: ArrivalConfig) -> ArrivalPattern:
    """Instantiate a single arrival pattern by name."""
    common = {"rate": cfg.rate, "seed": cfg.seed}
    if name == "Uniform":
        return UniformArrival(**common)
    if name == "Poisson":
        return PoissonArrival(**common)
    if name == "Gamma":
        return GammaArrival(**common, shape=cfg.gamma_shape)
    if name == "Fixed":
        return FixedIntervalArrival(rate=cfg.rate)
    if name == "Bursty":
        return BurstyArrival(
            high_rate=cfg.bursty_high_rate,
            low_rate=cfg.bursty_low_rate,
            period=cfg.bursty_period,
            burst_ratio=cfg.bursty_burst_ratio,
            seed=cfg.seed,
        )
    msg = f"Unknown arrival pattern: {name!r}"
    raise ValueError(msg)


# -- Public suite interface ---------------------------------------------------


def build(
    profile: ProfileCache,
    sim_cfg: SimulationConfig,
    arrival_cfg: ArrivalConfig | None = None,
) -> list[ExperimentConfig]:
    """Build configs for every (admission, dispatcher) combination."""
    adm_exp_cfg = AdmissionExperimentConfig()
    arrival_cfg = arrival_cfg or ArrivalConfig()
    pattern = _make_arrival_pattern(adm_exp_cfg.arrival_pattern, arrival_cfg)
    arrival_times = tuple(pattern.generate(sim_cfg.duration))

    configs: list[ExperimentConfig] = []
    for adm_spec, admission in build_admissions():
        for spec, dispatcher in build_dispatchers(profile.available_models):
            configs.append(
                ExperimentConfig(
                    name=f"{adm_spec.name}{_SEP}{spec.name}",
                    description=spec.description,
                    simulation=sim_cfg,
                    admission=admission,
                    dispatcher=dispatcher,
                    arrival_times=arrival_times,
                )
            )
    return configs


def report(
    results: dict[str, MetricsSummary],
    output_dir: Path,
    *,
    plot_config: PlotConfig | None = None,
    arrival_cfg: ArrivalConfig | None = None,
    per_run: dict[str, list[MetricsSummary]] | None = None,
) -> None:
    """Generate the admission Typst table and tradeoff plot."""
    suite_dir = output_dir / "admission"
    suite_dir.mkdir(parents=True, exist_ok=True)

    adm_cfg = AdmissionExperimentConfig()
    (suite_dir / "results_table.typ").write_text(
        generate_admission_typst_table(results, adm_cfg=adm_cfg)
    )
    generate_admission_tradeoff_plot(results, suite_dir, config=plot_config)


def console_columns() -> tuple[_Column, ...] | None:
    """Return the column subset for the console table."""
    adm_cfg = AdmissionExperimentConfig()
    return select_columns(adm_cfg.table_metrics)

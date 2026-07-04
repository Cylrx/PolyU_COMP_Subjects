"""Dispatcher comparison suite.

Tests all dispatch strategies against multiple arrival patterns.
Admission is fixed to DropOld; the two axes are dispatcher × arrival pattern.
"""

from __future__ import annotations

from pathlib import Path

from config import ArrivalConfig, PlotConfig, SimulationConfig
from core.types import ModelVariant, ProfileCache
from evaluation.analysis import generate_all_plots, generate_queue_timeline
from evaluation.metrics import MetricsSummary
from evaluation.typst_table import generate_typst_table
from experiments._dispatchers import build_dispatchers
from experiments.runner import ExperimentConfig
from strategies.admission import DropOld
from strategies.arrival import BurstyArrival, FixedIntervalArrival, GammaArrival, PoissonArrival, UniformArrival
from strategies.protocols import ArrivalPattern

_SEP = " | "

def _build_arrival_patterns(cfg: ArrivalConfig) -> list[tuple[str, ArrivalPattern]]:
    """Build all arrival patterns from the shared config."""
    return [
        ("Uniform", UniformArrival(rate=cfg.rate, seed=cfg.seed)),
        ("Poisson", PoissonArrival(rate=cfg.rate, seed=cfg.seed)),
        ("Gamma", GammaArrival(rate=cfg.rate, seed=cfg.seed, shape=cfg.gamma_shape)),
        ("Fixed", FixedIntervalArrival(rate=cfg.rate)),
        ("Bursty", BurstyArrival(
            high_rate=cfg.bursty_high_rate,
            low_rate=cfg.bursty_low_rate,
            period=cfg.bursty_period,
            burst_ratio=cfg.bursty_burst_ratio,
            seed=cfg.seed,
        )),
    ]


# Default set used by the experiment grid (matches original 3-pattern suite).
ARRIVAL_PATTERNS: list[tuple[str, ArrivalPattern]] = _build_arrival_patterns(ArrivalConfig())[:3]


def _make_dispatch_configs(
    available: list[ModelVariant],
    sim_cfg: SimulationConfig,
    arrival_times: tuple[float, ...],
) -> list[ExperimentConfig]:
    """Build the 6 dispatcher configs for a given set of arrival times."""
    return [
        ExperimentConfig(
            name=f"DropOld + {spec.name}",
            description=spec.description,
            simulation=sim_cfg,
            admission=DropOld(),
            dispatcher=dispatcher,
            arrival_times=arrival_times,
        )
        for spec, dispatcher in build_dispatchers(available)
    ]


def build(
    profile: ProfileCache,
    sim_cfg: SimulationConfig,
    arrival_cfg: ArrivalConfig | None = None,
) -> list[ExperimentConfig]:
    """Build configs for every (arrival pattern, dispatcher) combination."""
    available = profile.available_models
    patterns = _build_arrival_patterns(arrival_cfg)[:3] if arrival_cfg else ARRIVAL_PATTERNS
    configs: list[ExperimentConfig] = []

    for pattern_name, pattern in patterns:
        arrival_times = tuple(pattern.generate(sim_cfg.duration))
        for cfg in _make_dispatch_configs(available, sim_cfg, arrival_times):
            configs.append(
                ExperimentConfig(
                    name=f"{pattern_name}{_SEP}{cfg.name}",
                    description=cfg.description,
                    simulation=cfg.simulation,
                    admission=cfg.admission,
                    dispatcher=cfg.dispatcher,
                    arrival_times=cfg.arrival_times,
                )
            )

    return configs


def report(
    results: dict[str, MetricsSummary],
    output_dir: Path,
    plot_config: PlotConfig | None = None,
    arrival_cfg: ArrivalConfig | None = None,
    per_run: dict[str, list[MetricsSummary]] | None = None,
) -> None:
    """Generate per-arrival-pattern plots and Typst results table."""
    suite_dir = output_dir / "dispatch"
    suite_dir.mkdir(parents=True, exist_ok=True)

    (suite_dir / "results_table.typ").write_text(
        generate_typst_table(results, arrival_cfg=arrival_cfg)
    )

    patterns = _build_arrival_patterns(arrival_cfg)[:3] if arrival_cfg else ARRIVAL_PATTERNS

    for pattern_name, _ in patterns:
        prefix = f"{pattern_name}{_SEP}"
        pattern_results = {
            k.removeprefix(prefix): v
            for k, v in results.items()
            if k.startswith(prefix)
        }
        if pattern_results:
            pattern_dir = suite_dir / pattern_name.lower()
            pattern_dir.mkdir(parents=True, exist_ok=True)
            generate_all_plots(pattern_results, pattern_dir, config=plot_config)

    if per_run:
        cfg = plot_config if plot_config is not None else PlotConfig()
        pattern_name = cfg.queue_timeline_pattern
        prefix = f"{pattern_name}{_SEP}"
        first_run = {
            k.removeprefix(prefix): summaries[0]
            for k, summaries in per_run.items()
            if k.startswith(prefix)
        }
        if first_run:
            generate_queue_timeline(
                first_run, pattern_name, suite_dir, config=plot_config,
            )

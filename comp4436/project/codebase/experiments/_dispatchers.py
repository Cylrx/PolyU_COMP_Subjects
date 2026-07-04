"""Shared dispatcher specifications for experiment suites.

Single source of truth for the 6 dispatch strategies used across
experiment grids and result tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import MultiStepOptimalConfig, OneStepOptimalConfig
from strategies.dispatch import (
    AdaptiveDeadlineDispatch,
    AdaptiveQueueDispatch,
    FixedEdgeDispatch,
    FixedOffloadDispatch,
    MultiStepOptimalDispatch,
    OneStepOptimalDispatch,
)

if TYPE_CHECKING:
    from core.types import ModelVariant
    from strategies.protocols import DispatchStrategy


@dataclass(frozen=True)
class DispatcherSpec:
    """Metadata for one dispatcher in the experiment grid."""

    name: str
    description: str
    abbreviation: str
    is_baseline: bool
    header_bold: bool


SPECS: tuple[DispatcherSpec, ...] = (
    DispatcherSpec("FixedEdge", "Edge-only baseline", "FxE", is_baseline=True, header_bold=False),
    DispatcherSpec("FixedOffload", "Offload baseline", "FxO", is_baseline=True, header_bold=False),
    DispatcherSpec("AdaptiveQueue", "Queue-adaptive dispatch", "AdQ", is_baseline=False, header_bold=False),
    DispatcherSpec("AdaptiveDeadline", "Deadline-aware dispatch", "AdD", is_baseline=False, header_bold=False),
    DispatcherSpec("OneStepOptimal", "One-step optimal (Lyapunov)", "1SO", is_baseline=False, header_bold=True),
    DispatcherSpec("MultiStepOptimal", "Multi-step optimal (DDP)", "MSO", is_baseline=False, header_bold=True),
)


def build_dispatchers(
    available: list[ModelVariant],
) -> list[tuple[DispatcherSpec, DispatchStrategy]]:
    """Create all 6 dispatcher instances.

    Returns ``(spec, instance)`` pairs in the canonical order defined
    by :data:`SPECS`.
    """
    default_model = available[0].name
    os_cfg = OneStepOptimalConfig()
    ms_cfg = MultiStepOptimalConfig()

    instances: dict[str, DispatchStrategy] = {
        "FixedEdge": FixedEdgeDispatch(default_model),
        "FixedOffload": FixedOffloadDispatch(default_model),
        "AdaptiveQueue": AdaptiveQueueDispatch(),
        "AdaptiveDeadline": AdaptiveDeadlineDispatch(),
        "OneStepOptimal": OneStepOptimalDispatch(
            value_scale_multiplier=os_cfg.value_scale_multiplier,
            backlog_weight=os_cfg.backlog_weight,
        ),
        "MultiStepOptimal": MultiStepOptimalDispatch(
            max_backlog_window=ms_cfg.max_backlog_window,
            time_quantum=ms_cfg.time_quantum,
            service_time_penalty=ms_cfg.service_time_penalty,
        ),
    }
    return [(spec, instances[spec.name]) for spec in SPECS]

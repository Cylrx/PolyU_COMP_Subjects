"""Shared admission-policy specifications for experiment suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategies.admission import DropOld, DropTail

if TYPE_CHECKING:
    from strategies.protocols import AdmissionStrategy


@dataclass(frozen=True)
class AdmissionSpec:
    """Metadata for one admission policy in the experiment grid."""

    name: str
    abbreviation: str


SPECS: tuple[AdmissionSpec, ...] = (
    AdmissionSpec("DropOld", "DrO"),
    AdmissionSpec("DropTail", "DrT"),
)


def abbreviation_for(name: str) -> str:
    """Return the configured abbreviation for *name* if available."""
    for spec in SPECS:
        if spec.name == name:
            return spec.abbreviation
    return name


def build_admissions() -> list[tuple[AdmissionSpec, AdmissionStrategy]]:
    """Create all configured admission-policy instances in canonical order."""
    instances: dict[str, AdmissionStrategy] = {
        "DropOld": DropOld(),
        "DropTail": DropTail(),
    }
    return [(spec, instances[spec.name]) for spec in SPECS]

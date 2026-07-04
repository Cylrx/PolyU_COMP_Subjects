from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from core.queue import ProcessingQueue
    from core.types import DataSample, Location, ModelVariant


class AdmissionResult(Enum):
    ADMIT = auto()
    REJECT = auto()
    ADMIT_EVICT_OLDEST = auto()


class AdmissionStrategy(Protocol):
    def evaluate(
        self,
        sample: DataSample,
        queue: ProcessingQueue,
        current_time: float,
    ) -> AdmissionResult: ...


class DispatchAction(Enum):
    DISPATCH = auto()
    WAIT = auto()
    DROP = auto()


@dataclass(frozen=True)
class DispatchContext:
    current_time: float
    queue_capacity: int
    queued_samples: tuple[DataSample, ...]
    available_models: tuple[ModelVariant, ...]
    edge_available: bool
    edge_next_available_time: float
    edge_loaded_model: str | None
    edge_model_load_time: float
    cloud_available: bool
    cloud_next_available_time: float
    expected_cloud_rtt: float


@dataclass(frozen=True)
class DispatchDecision:
    action: DispatchAction
    model_name: str | None = None
    location: Location | None = None


class DispatchStrategy(Protocol):
    def decide(
        self,
        sample: DataSample,
        context: DispatchContext,
    ) -> DispatchDecision: ...


class ArrivalPattern(Protocol):
    def generate(self, duration: float) -> list[float]:
        """Return a sorted list of arrival times within [0, duration)."""
        ...

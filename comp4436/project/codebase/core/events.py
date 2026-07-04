from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventKind(Enum):
    SAMPLE_ARRIVAL = auto()
    INFERENCE_COMPLETE = auto()
    MODEL_LOAD_COMPLETE = auto()


@dataclass(order=True)
class Event:
    """A single simulation event, ordered by (time, sequence) for heap use."""

    time: float
    sequence: int
    kind: EventKind = field(compare=False)
    payload: Any = field(compare=False, repr=False)

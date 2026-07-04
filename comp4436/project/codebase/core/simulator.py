from __future__ import annotations

import heapq
from typing import Any, Callable

from core.events import Event, EventKind


class Simulator:
    """Discrete event-driven simulation engine.

    Events are stored in a min-heap ordered by (time, sequence).
    The clock jumps from event to event — no wasted ticks.
    """

    def __init__(self) -> None:
        self.clock: float = 0.0
        self._events: list[Event] = []
        self._sequence: int = 0
        self._handlers: dict[EventKind, Callable[[Event], None]] = {}

    def register(self, kind: EventKind, handler: Callable[[Event], None]) -> None:
        """Register a handler for a specific event kind."""
        self._handlers[kind] = handler

    def schedule(self, time: float, kind: EventKind, payload: Any = None) -> None:
        """Schedule an event at a future (or current) time."""
        if time < self.clock:
            raise ValueError(
                f"Cannot schedule event in the past: {time} < {self.clock}"
            )
        event = Event(
            time=time,
            sequence=self._sequence,
            kind=kind,
            payload=payload,
        )
        self._sequence += 1
        heapq.heappush(self._events, event)

    def run(self) -> None:
        """Drain the event queue, dispatching each event to its handler."""
        while self._events:
            event = heapq.heappop(self._events)
            self.clock = event.time
            handler = self._handlers.get(event.kind)
            if handler is None:
                raise RuntimeError(f"No handler registered for {event.kind}")
            handler(event)

    @property
    def pending_count(self) -> int:
        return len(self._events)

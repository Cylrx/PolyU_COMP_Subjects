from __future__ import annotations

import heapq

from core.network import NetworkModel


class CloudNode:
    """Simulates a cloud server with bounded concurrency.

    Inference is faster than edge (no slowdown multiplier), but incurs
    network round-trip latency.
    """

    def __init__(self, capacity: int, network: NetworkModel) -> None:
        self.capacity = capacity
        self.network = network
        self._active_until: list[float] = []

    def is_available(self) -> bool:
        return len(self._active_until) < self.capacity

    @property
    def active_count(self) -> int:
        return len(self._active_until)

    @property
    def next_available_time(self) -> float:
        if self.is_available():
            return 0.0
        return self._active_until[0]

    def start_request(self, end_time: float) -> None:
        if not self.is_available():
            raise RuntimeError("Cloud has no free execution slot")
        heapq.heappush(self._active_until, end_time)

    def finish_request(self, end_time: float) -> None:
        try:
            idx = self._active_until.index(end_time)
        except ValueError as exc:
            raise RuntimeError(
                f"Cloud completion time {end_time} was not reserved"
            ) from exc

        last = self._active_until.pop()
        if idx < len(self._active_until):
            self._active_until[idx] = last
            heapq.heapify(self._active_until)

    def compute_latency(self, profiled_latency: float) -> float:
        """Cloud inference time = profiled latency + network round-trip."""
        return profiled_latency + self.network.round_trip_latency()

from __future__ import annotations

import random


class NetworkModel:
    """Simulates network latency between edge and cloud.

    Uses a private RNG instance so results are reproducible and independent
    of global random state.
    """

    def __init__(
        self,
        base_rtt: float,
        jitter_std: float,
        seed: int = 0,
    ) -> None:
        self.base_rtt = base_rtt
        self.jitter_std = jitter_std
        self._rng = random.Random(seed)

    def one_way_latency(self) -> float:
        """Sample a single one-way latency (half RTT + jitter)."""
        return max(0.0, self.base_rtt / 2 + self._rng.gauss(0.0, self.jitter_std))

    def round_trip_latency(self) -> float:
        """Sample a full round-trip latency (upload + download)."""
        return self.one_way_latency() + self.one_way_latency()

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod


class _StochasticArrival(ABC):
    """Base class for renewal processes with IID inter-arrival times."""

    def __init__(self, rate: float, seed: int = 0) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self._rng = random.Random(seed)

    def generate(self, duration: float) -> list[float]:
        if duration <= 0:
            return []

        times: list[float] = []
        t = 0.0
        while True:
            interval = self._sample_interval()
            if interval <= 0:
                interval = math.nextafter(0.0, math.inf)
            t += interval
            if t >= duration:
                break
            times.append(t)
        return times

    @abstractmethod
    def _sample_interval(self) -> float:
        """Return the next inter-arrival time."""


class PoissonArrival(_StochasticArrival):
    """Arrivals follow a Poisson process (exponential inter-arrival times)."""

    def _sample_interval(self) -> float:
        return self._rng.expovariate(self.rate)


class UniformArrival(_StochasticArrival):
    """Arrivals with uniformly distributed inter-arrival times.

    Inter-arrival times follow ``Uniform(0, 2 / rate)`` so the mean
    inter-arrival time remains ``1 / rate``.
    """

    def _sample_interval(self) -> float:
        return self._rng.uniform(0.0, 2.0 / self.rate)


class FixedIntervalArrival:
    """Arrivals at exact uniform intervals (1/rate seconds apart)."""

    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate

    def generate(self, duration: float) -> list[float]:
        if duration <= 0:
            return []

        interval = 1.0 / self.rate
        times: list[float] = []
        t = interval
        while t < duration:
            times.append(t)
            t += interval
        return times


class GammaArrival(_StochasticArrival):
    """Arrivals with gamma-distributed inter-arrival times.

    ``shape`` defaults to the Proteus paper's highly bursty setting.
    The scale is derived so that the mean inter-arrival time remains
    ``1 / rate``.
    """

    def __init__(self, rate: float, seed: int = 0, shape: float = 0.1) -> None:
        if shape <= 0:
            raise ValueError(f"shape must be positive, got {shape}")
        super().__init__(rate=rate, seed=seed)
        self.shape = shape
        self._scale = 1.0 / (self.rate * self.shape)

    def _sample_interval(self) -> float:
        return self._rng.gammavariate(alpha=self.shape, beta=self._scale)


class BurstyArrival:
    """Alternates between high-rate bursts and low-rate calm periods.

    Each cycle lasts `period` seconds. The first `burst_ratio` fraction
    of each cycle uses `high_rate`; the remainder uses `low_rate`.
    Both sub-periods use Poisson arrivals internally.
    """

    def __init__(
        self,
        high_rate: float,
        low_rate: float,
        period: float = 5.0,
        burst_ratio: float = 0.4,
        seed: int = 0,
    ) -> None:
        if high_rate <= 0:
            raise ValueError(f"high_rate must be positive, got {high_rate}")
        if low_rate <= 0:
            raise ValueError(f"low_rate must be positive, got {low_rate}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        if not 0 < burst_ratio < 1:
            raise ValueError(
                f"burst_ratio must be between 0 and 1, got {burst_ratio}"
            )
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.period = period
        self.burst_ratio = burst_ratio
        self._rng = random.Random(seed)

    def generate(self, duration: float) -> list[float]:
        if duration <= 0:
            return []

        times: list[float] = []
        cycle_start = 0.0

        while cycle_start < duration:
            burst_end = cycle_start + self.period * self.burst_ratio
            cycle_end = cycle_start + self.period

            # Burst phase
            t = cycle_start
            while True:
                t += self._rng.expovariate(self.high_rate)
                if t >= min(burst_end, duration):
                    break
                times.append(t)

            # Calm phase
            t = burst_end
            while True:
                t += self._rng.expovariate(self.low_rate)
                if t >= min(cycle_end, duration):
                    break
                times.append(t)

            cycle_start = cycle_end

        return times

from __future__ import annotations

import pytest

from strategies.arrival import (
    FixedIntervalArrival,
    GammaArrival,
    PoissonArrival,
    UniformArrival,
)


def _assert_sorted_and_bounded(times: list[float], duration: float) -> None:
    assert times == sorted(times)
    assert all(0.0 <= time < duration for time in times)


@pytest.mark.parametrize(
    ("factory", "duration"),
    [
        (lambda: PoissonArrival(rate=10.0, seed=7), 5.0),
        (lambda: UniformArrival(rate=10.0, seed=7), 5.0),
        (lambda: GammaArrival(rate=10.0, seed=7), 5.0),
    ],
)
def test_stochastic_arrivals_are_seed_deterministic(factory, duration: float) -> None:
    a = factory().generate(duration)
    b = factory().generate(duration)

    assert a == b
    _assert_sorted_and_bounded(a, duration)


@pytest.mark.parametrize(
    "arrival_cls",
    [PoissonArrival, UniformArrival, GammaArrival],
)
def test_stochastic_arrivals_change_with_seed(arrival_cls: type) -> None:
    a = arrival_cls(rate=10.0, seed=1).generate(5.0)
    b = arrival_cls(rate=10.0, seed=2).generate(5.0)

    assert a != b


@pytest.mark.parametrize(
    "pattern",
    [
        PoissonArrival(rate=10.0, seed=0),
        UniformArrival(rate=10.0, seed=0),
        GammaArrival(rate=10.0, seed=0),
        FixedIntervalArrival(rate=10.0),
    ],
)
@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_generate_returns_empty_for_nonpositive_duration(pattern, duration: float) -> None:
    assert pattern.generate(duration) == []


def test_fixed_interval_arrival_remains_periodic() -> None:
    times = FixedIntervalArrival(rate=4.0).generate(1.1)

    assert times == [0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize(
    ("arrival_cls", "kwargs"),
    [
        (PoissonArrival, {"rate": 0.0}),
        (PoissonArrival, {"rate": -1.0}),
        (UniformArrival, {"rate": 0.0}),
        (UniformArrival, {"rate": -1.0}),
        (GammaArrival, {"rate": 0.0}),
        (GammaArrival, {"rate": -1.0}),
        (GammaArrival, {"rate": 1.0, "shape": 0.0}),
        (GammaArrival, {"rate": 1.0, "shape": -0.1}),
        (FixedIntervalArrival, {"rate": 0.0}),
        (FixedIntervalArrival, {"rate": -1.0}),
    ],
)
def test_invalid_constructor_parameters_raise_value_error(
    arrival_cls: type,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        arrival_cls(**kwargs)


@pytest.mark.parametrize(
    ("pattern", "rate", "duration", "rel_tolerance"),
    [
        (UniformArrival(rate=25.0, seed=11), 25.0, 4000.0, 0.03),
        (PoissonArrival(rate=25.0, seed=11), 25.0, 4000.0, 0.05),
        (GammaArrival(rate=25.0, seed=11), 25.0, 4000.0, 0.08),
    ],
)
def test_stochastic_arrivals_match_target_rate_on_long_runs(
    pattern,
    rate: float,
    duration: float,
    rel_tolerance: float,
) -> None:
    realized = len(pattern.generate(duration))
    expected = rate * duration

    assert realized == pytest.approx(expected, rel=rel_tolerance)


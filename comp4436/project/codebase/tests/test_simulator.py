from core.events import EventKind
from core.simulator import Simulator

import pytest


class TestSimulator:
    def test_events_dispatch_in_time_order(self) -> None:
        sim = Simulator()
        order: list[float] = []
        sim.register(EventKind.SAMPLE_ARRIVAL, lambda e: order.append(e.time))

        sim.schedule(0.3, EventKind.SAMPLE_ARRIVAL)
        sim.schedule(0.1, EventKind.SAMPLE_ARRIVAL)
        sim.schedule(0.2, EventKind.SAMPLE_ARRIVAL)
        sim.run()

        assert order == [0.1, 0.2, 0.3]

    def test_tie_breaking_by_sequence(self) -> None:
        sim = Simulator()
        payloads: list[str] = []
        sim.register(
            EventKind.SAMPLE_ARRIVAL, lambda e: payloads.append(e.payload)
        )

        sim.schedule(1.0, EventKind.SAMPLE_ARRIVAL, "first")
        sim.schedule(1.0, EventKind.SAMPLE_ARRIVAL, "second")
        sim.schedule(1.0, EventKind.SAMPLE_ARRIVAL, "third")
        sim.run()

        assert payloads == ["first", "second", "third"]

    def test_clock_advances_correctly(self) -> None:
        sim = Simulator()
        clocks: list[float] = []
        sim.register(
            EventKind.SAMPLE_ARRIVAL, lambda e: clocks.append(sim.clock)
        )

        sim.schedule(0.5, EventKind.SAMPLE_ARRIVAL)
        sim.schedule(1.5, EventKind.SAMPLE_ARRIVAL)
        sim.run()

        assert clocks == [0.5, 1.5]
        assert sim.clock == 1.5

    def test_scheduling_in_the_past_raises(self) -> None:
        sim = Simulator()
        sim.register(EventKind.SAMPLE_ARRIVAL, lambda e: None)
        sim.schedule(1.0, EventKind.SAMPLE_ARRIVAL)
        sim.run()

        with pytest.raises(ValueError, match="past"):
            sim.schedule(0.5, EventKind.SAMPLE_ARRIVAL)

    def test_scheduling_from_within_handler(self) -> None:
        sim = Simulator()
        results: list[str] = []

        def on_arrival(event: object) -> None:
            results.append("arrival")
            sim.schedule(sim.clock + 0.1, EventKind.INFERENCE_COMPLETE)

        def on_complete(event: object) -> None:
            results.append("complete")

        sim.register(EventKind.SAMPLE_ARRIVAL, on_arrival)
        sim.register(EventKind.INFERENCE_COMPLETE, on_complete)
        sim.schedule(0.0, EventKind.SAMPLE_ARRIVAL)
        sim.run()

        assert results == ["arrival", "complete"]
        assert sim.clock == pytest.approx(0.1)

    def test_unregistered_handler_raises(self) -> None:
        sim = Simulator()
        sim.schedule(0.0, EventKind.SAMPLE_ARRIVAL)

        with pytest.raises(RuntimeError, match="No handler"):
            sim.run()

    def test_pending_count(self) -> None:
        sim = Simulator()
        sim.register(EventKind.SAMPLE_ARRIVAL, lambda e: None)

        assert sim.pending_count == 0
        sim.schedule(0.0, EventKind.SAMPLE_ARRIVAL)
        sim.schedule(1.0, EventKind.SAMPLE_ARRIVAL)
        assert sim.pending_count == 2

        sim.run()
        assert sim.pending_count == 0

    def test_empty_simulator_runs_without_error(self) -> None:
        sim = Simulator()
        sim.run()
        assert sim.clock == 0.0

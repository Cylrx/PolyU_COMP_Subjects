from core.edge import EdgeNode

import pytest


class TestEdgeNode:
    def test_is_available_when_not_busy(self) -> None:
        edge = EdgeNode()
        assert edge.is_available()

    def test_not_available_when_busy(self) -> None:
        edge = EdgeNode()
        edge.busy = True
        assert not edge.is_available()

    def test_needs_model_switch_when_no_model_loaded(self) -> None:
        edge = EdgeNode()
        assert edge.needs_model_switch("full")

    def test_needs_model_switch_when_different_model(self) -> None:
        edge = EdgeNode()
        edge.loaded_model = "full"
        assert edge.needs_model_switch("quantized")

    def test_no_switch_needed_when_same_model(self) -> None:
        edge = EdgeNode()
        edge.loaded_model = "full"
        assert not edge.needs_model_switch("full")

    def test_reserve_until_marks_edge_busy(self) -> None:
        edge = EdgeNode()
        edge.reserve_until(0.25)

        assert edge.busy
        assert edge.next_available_time == pytest.approx(0.25)

    def test_compute_latency_uses_profile_directly(self) -> None:
        edge = EdgeNode()
        assert edge.compute_latency(0.01) == pytest.approx(0.01)

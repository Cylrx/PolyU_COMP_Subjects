from core.cloud import CloudNode
from core.network import NetworkModel

import pytest


class TestCloudNode:
    def test_is_available_under_capacity(self) -> None:
        net = NetworkModel(base_rtt=0.1, jitter_std=0.0, seed=0)
        cloud = CloudNode(capacity=3, network=net)

        assert cloud.is_available()
        cloud.start_request(0.2)
        cloud.start_request(0.3)
        assert cloud.is_available()

    def test_not_available_at_capacity(self) -> None:
        net = NetworkModel(base_rtt=0.1, jitter_std=0.0, seed=0)
        cloud = CloudNode(capacity=3, network=net)
        cloud.start_request(0.2)
        cloud.start_request(0.3)
        cloud.start_request(0.4)

        assert not cloud.is_available()

    def test_next_available_time_tracks_earliest_completion(self) -> None:
        net = NetworkModel(base_rtt=0.1, jitter_std=0.0, seed=0)
        cloud = CloudNode(capacity=1, network=net)
        cloud.start_request(0.4)

        assert cloud.next_available_time == pytest.approx(0.4)

        cloud.finish_request(0.4)
        assert cloud.is_available()

    def test_compute_latency_includes_network(self) -> None:
        # Zero jitter so network RTT is deterministic = base_rtt
        net = NetworkModel(base_rtt=0.1, jitter_std=0.0, seed=0)
        cloud = CloudNode(capacity=3, network=net)

        latency = cloud.compute_latency(profiled_latency=0.01)
        assert latency == pytest.approx(0.11)  # 0.01 + 0.1 RTT

    def test_compute_latency_varies_with_jitter(self) -> None:
        net = NetworkModel(base_rtt=0.1, jitter_std=0.02, seed=42)
        cloud = CloudNode(capacity=3, network=net)

        latencies = [cloud.compute_latency(0.01) for _ in range(20)]
        # All should be positive
        assert all(lat > 0 for lat in latencies)
        # With jitter, not all latencies should be identical
        assert len(set(round(lat, 6) for lat in latencies)) > 1


class TestNetworkModel:
    def test_zero_jitter_gives_deterministic_rtt(self) -> None:
        net = NetworkModel(base_rtt=0.2, jitter_std=0.0, seed=0)

        assert net.round_trip_latency() == pytest.approx(0.2)
        assert net.one_way_latency() == pytest.approx(0.1)

    def test_latency_never_negative(self) -> None:
        # High jitter relative to base to stress-test the max(0, ...) guard.
        net = NetworkModel(base_rtt=0.01, jitter_std=0.1, seed=99)

        for _ in range(100):
            assert net.one_way_latency() >= 0.0
            assert net.round_trip_latency() >= 0.0

    def test_reproducible_with_same_seed(self) -> None:
        net1 = NetworkModel(base_rtt=0.1, jitter_std=0.02, seed=7)
        net2 = NetworkModel(base_rtt=0.1, jitter_std=0.02, seed=7)

        for _ in range(10):
            assert net1.round_trip_latency() == net2.round_trip_latency()

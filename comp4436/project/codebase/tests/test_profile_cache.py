from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.types import ProfileCache, ProfileEntry


class TestProfileCacheJSON:
    def test_save_load_round_trip(self) -> None:
        original = ProfileCache.mock(n_samples=50, seed=99)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_profile.json"
            original.save(path)
            loaded = ProfileCache.load(path)

        assert loaded.size == original.size
        assert loaded.model_names == original.model_names

        for idx in range(50):
            for name in original.model_names:
                orig = original.lookup(idx, name)
                load = loaded.lookup(idx, name)
                assert orig.prediction == load.prediction
                assert orig.label == load.label
                assert orig.correct == load.correct
                assert abs(orig.edge_latency - load.edge_latency) < 1e-10
                assert abs(orig.cloud_latency - load.cloud_latency) < 1e-10

    def test_metadata_preserved(self) -> None:
        original = ProfileCache.mock(n_samples=20, seed=7)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            original.save(path)
            loaded = ProfileCache.load(path)

        assert loaded.metadata.kind == "mock"
        assert loaded.metadata.n_samples == 20
        assert loaded.metadata.edge_device == "mock-cpu"
        assert loaded.metadata.cloud_device == "mock-cuda"
        assert loaded.metadata.torch_version is None
        assert loaded.metadata.model_variants == original.metadata.model_variants

    def test_available_models_from_model_info(self) -> None:
        profile = ProfileCache.mock(n_samples=10)
        models = profile.available_models

        assert len(models) == 6
        names = {m.name for m in models}
        assert "full" in names
        assert "structured_30" in names

        for m in models:
            assert m.accuracy > 0
            assert m.edge_avg_latency > 0
            assert m.cloud_avg_latency > 0
            assert m.macs > 0

    def test_mock_returns_valid_structure(self) -> None:
        profile = ProfileCache.mock(n_samples=5, model_names=["a", "b"])

        assert profile.size == 5
        assert profile.model_names == ["a", "b"]
        assert profile.metadata.n_samples == 5

        entry = profile.lookup(0, "a")
        assert isinstance(entry, ProfileEntry)
        assert 0 <= entry.prediction <= 9

    def test_json_file_is_human_readable(self) -> None:
        profile = ProfileCache.mock(n_samples=3, model_names=["full"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            profile.save(path)
            content = path.read_text()

        assert '"metadata"' in content
        assert '"model_info"' in content
        assert '"entries"' in content
        assert '"0,full"' in content

    def test_old_schema_is_rejected(self) -> None:
        legacy_doc = {
            "metadata": {
                "kind": "profiled",
                "n_samples": 1,
                "model_variants": ["full"],
                "device": "cpu",
                "description": "legacy",
                "created_at": "2026-04-04T00:00:00+00:00",
                "torch_version": None,
            },
            "model_info": {
                "full": {
                    "name": "full",
                    "accuracy": 0.9,
                    "avg_latency": 0.01,
                    "macs": 1,
                }
            },
            "entries": {
                "0,full": {
                    "prediction": 0,
                    "label": 0,
                    "correct": True,
                    "latency": 0.01,
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(legacy_doc))
            with pytest.raises(ValueError, match="Unsupported profile schema"):
                ProfileCache.load(path)

    def test_mock_model_stats_match_realized_samples(self) -> None:
        profile = ProfileCache.mock(n_samples=200, seed=1)

        for model in profile.available_models:
            correctness = [
                profile.lookup(idx, model.name).correct
                for idx in range(profile.size)
            ]
            edge_latencies = [
                profile.lookup(idx, model.name).edge_latency
                for idx in range(profile.size)
            ]
            cloud_latencies = [
                profile.lookup(idx, model.name).cloud_latency
                for idx in range(profile.size)
            ]

            assert model.accuracy == pytest.approx(sum(correctness) / len(correctness))
            assert model.edge_avg_latency == pytest.approx(
                sum(edge_latencies) / len(edge_latencies)
            )
            assert model.cloud_avg_latency == pytest.approx(
                sum(cloud_latencies) / len(cloud_latencies)
            )

    def test_mock_has_heterogeneous_device_ordering(self) -> None:
        profile = ProfileCache.mock(n_samples=200, seed=3)
        by_name = {model.name: model for model in profile.available_models}

        # Quantization excels on CPU but hurts GPU; structured pruning helps both.
        assert by_name["quantized_int8"].edge_avg_latency < by_name["structured_50"].edge_avg_latency
        assert by_name["quantized_int8"].cloud_avg_latency > by_name["structured_50"].cloud_avg_latency

    def test_mock_latency_jitter_stays_small(self) -> None:
        profile = ProfileCache.mock(n_samples=200, seed=5)

        for model in profile.available_models:
            edge_latencies = [
                profile.lookup(idx, model.name).edge_latency
                for idx in range(profile.size)
            ]
            cloud_latencies = [
                profile.lookup(idx, model.name).cloud_latency
                for idx in range(profile.size)
            ]
            edge_spread = max(edge_latencies) / min(edge_latencies)
            cloud_spread = max(cloud_latencies) / min(cloud_latencies)

            assert edge_spread < 1.07
            assert cloud_spread < 1.07

    def test_mock_latency_scale_factors_scale_entries_and_averages(self) -> None:
        baseline = ProfileCache.mock(
            n_samples=50,
            seed=11,
            edge_latency_scale_factor=1.0,
            cloud_latency_scale_factor=1.0,
        )
        scaled = ProfileCache.mock(
            n_samples=50,
            seed=11,
            edge_latency_scale_factor=2.5,
            cloud_latency_scale_factor=4.0,
        )

        for model_name in baseline.model_names:
            baseline_model = next(m for m in baseline.available_models if m.name == model_name)
            scaled_model = next(m for m in scaled.available_models if m.name == model_name)

            assert scaled_model.edge_avg_latency == pytest.approx(
                baseline_model.edge_avg_latency * 2.5
            )
            assert scaled_model.cloud_avg_latency == pytest.approx(
                baseline_model.cloud_avg_latency * 4.0
            )

            baseline_entry = baseline.lookup(0, model_name)
            scaled_entry = scaled.lookup(0, model_name)
            assert scaled_entry.edge_latency == pytest.approx(
                baseline_entry.edge_latency * 2.5
            )
            assert scaled_entry.cloud_latency == pytest.approx(
                baseline_entry.cloud_latency * 4.0
            )

    def test_blacklisted_profile_hides_removed_variants_everywhere(self) -> None:
        profile = ProfileCache.mock(n_samples=5, seed=13)

        filtered = profile.blacklisted(("structured_50",))

        assert "structured_50" not in filtered.model_names
        assert "structured_50" not in filtered.metadata.model_variants
        assert all(model.name != "structured_50" for model in filtered.available_models)
        with pytest.raises(KeyError):
            filtered.lookup(0, "structured_50")

    def test_blacklisted_profile_rejects_removing_every_variant(self) -> None:
        profile = ProfileCache.mock(n_samples=5, model_names=["full"])

        with pytest.raises(ValueError, match="removed every variant"):
            profile.blacklisted(("full",))

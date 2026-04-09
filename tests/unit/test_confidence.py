import math
from datetime import datetime, timedelta, timezone

from memory_platform.ext.confidence import (
    compute_confidence,
    filter_by_confidence,
)


class TestComputeConfidence:
    def test_no_decay_for_fresh_memory(self):
        now = datetime.now(timezone.utc)
        result = compute_confidence(similarity=0.9, updated_at=now, layer="L1", now=now)
        assert result == 0.9

    def test_l1_slow_decay(self):
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(hours=168)
        result = compute_confidence(similarity=0.9, updated_at=week_ago, layer="L1", now=now)
        expected = 0.9 * math.exp(-0.001 * 168)
        assert abs(result - expected) < 0.001

    def test_l3_fast_decay(self):
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(hours=168)
        result = compute_confidence(similarity=0.9, updated_at=week_ago, layer="L3", now=now)
        expected = 0.9 * math.exp(-0.02 * 168)
        assert abs(result - expected) < 0.001

    def test_default_layer_is_l1(self):
        now = datetime.now(timezone.utc)
        result = compute_confidence(
            similarity=0.8, updated_at=now, layer=None, now=now
        )
        assert result == 0.8

    def test_updated_at_as_string(self):
        now = datetime.now(timezone.utc)
        updated_str = now.isoformat()
        result = compute_confidence(
            similarity=0.9, updated_at=updated_str, layer="L1", now=now
        )
        assert result == 0.9


class TestFilterByConfidence:
    def test_filters_below_threshold(self):
        now = datetime.now(timezone.utc)
        memories = [
            {"id": "1", "memory": "test", "score": 0.9, "metadata": {"memory_layer": "L1"},
             "updated_at": now.isoformat(), "created_at": now.isoformat()},
            {"id": "2", "memory": "old", "score": 0.9, "metadata": {"memory_layer": "L3"},
             "updated_at": (now - timedelta(hours=500)).isoformat(), "created_at": now.isoformat()},
        ]
        results = filter_by_confidence(memories, min_confidence=0.5, now=now)
        assert len(results) == 1
        assert results[0][0]["id"] == "1"

    def test_sorts_by_confidence_descending(self):
        now = datetime.now(timezone.utc)
        memories = [
            {"id": "1", "memory": "low", "score": 0.7, "metadata": {"memory_layer": "L2"},
             "updated_at": (now - timedelta(hours=100)).isoformat(), "created_at": now.isoformat()},
            {"id": "2", "memory": "high", "score": 0.9, "metadata": {"memory_layer": "L1"},
             "updated_at": now.isoformat(), "created_at": now.isoformat()},
        ]
        results = filter_by_confidence(memories, min_confidence=0.0, now=now)
        assert results[0][1] >= results[1][1]

    def test_respects_limit(self):
        now = datetime.now(timezone.utc)
        memories = [
            {"id": str(i), "memory": f"m{i}", "score": 0.9,
             "metadata": {"memory_layer": "L1"},
             "updated_at": now.isoformat(), "created_at": now.isoformat()}
            for i in range(10)
        ]
        results = filter_by_confidence(memories, min_confidence=0.0, limit=3, now=now)
        assert len(results) == 3

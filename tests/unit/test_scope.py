# tests/unit/test_scope.py
from memory_platform.ext.scope import Scope, apply_scope_filter, deduplicate_memories


class TestScope:
    def test_all_includes_shared_and_private(self):
        assert Scope.ALL.include_shared is True
        assert Scope.ALL.include_private is True

    def test_shared_only(self):
        assert Scope.SHARED.include_shared is True
        assert Scope.SHARED.include_private is False

    def test_private_only(self):
        assert Scope.PRIVATE.include_shared is False
        assert Scope.PRIVATE.include_private is True


class TestApplyScopeFilter:
    def test_all_returns_all(self):
        memories = [
            {"id": "1", "memory": "shared", "metadata": {"scope": "shared"}},
            {"id": "2", "memory": "private", "metadata": {"scope": "private"}},
        ]
        result = apply_scope_filter(memories, Scope.ALL)
        assert len(result) == 2

    def test_shared_filters_private(self):
        memories = [
            {"id": "1", "memory": "shared", "metadata": {"scope": "shared"}},
            {"id": "2", "memory": "private", "metadata": {"scope": "private"}},
        ]
        result = apply_scope_filter(memories, Scope.SHARED)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_private_filters_shared(self):
        memories = [
            {"id": "1", "memory": "shared", "metadata": {"scope": "shared"}},
            {"id": "2", "memory": "private", "metadata": {"scope": "private"}},
        ]
        result = apply_scope_filter(memories, Scope.PRIVATE)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_no_metadata_defaults_to_shared(self):
        memories = [
            {"id": "1", "memory": "no scope", "metadata": {}},
        ]
        result = apply_scope_filter(memories, Scope.SHARED)
        assert len(result) == 1


class TestDeduplicateMemories:
    def test_removes_duplicate_hashes(self):
        memories = [
            {"id": "1", "memory": "a", "hash": "h1"},
            {"id": "2", "memory": "a", "hash": "h1"},
            {"id": "3", "memory": "b", "hash": "h2"},
        ]
        result = deduplicate_memories(memories)
        assert len(result) == 2

    def test_no_duplicates(self):
        memories = [
            {"id": "1", "memory": "a", "hash": "h1"},
            {"id": "2", "memory": "b", "hash": "h2"},
        ]
        result = deduplicate_memories(memories)
        assert len(result) == 2

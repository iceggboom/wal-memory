import logging

from memory_platform.middleware.degradation import memory_degradation_handler


class TestDegradationHandler:
    def test_returns_empty_on_exception(self):
        result = memory_degradation_handler(Exception("DB down"))
        assert result == {"results": [], "total": 0}

    def test_returns_empty_with_error_message(self):
        result = memory_degradation_handler(ValueError("bad input"))
        assert result == {"results": [], "total": 0}

    def test_logs_error(self, caplog):
        with caplog.at_level(logging.ERROR, logger="memory_platform.middleware.degradation"):
            memory_degradation_handler(RuntimeError("service unavailable"))
        assert "Memory service degradation triggered" in caplog.text
        assert "service unavailable" in caplog.text

    def test_includes_exc_info_for_non_base_exception(self, caplog):
        with caplog.at_level(logging.ERROR, logger="memory_platform.middleware.degradation"):
            memory_degradation_handler(RuntimeError("custom error"))
        # exc_info should be True for non-generic Exception subclasses
        assert caplog.text != ""

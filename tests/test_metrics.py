"""Tests for observability metrics."""

from __future__ import annotations

from rag_pipeline.metrics import MetricsCollector, Timer, estimate_tokens


class TestMetricsCollector:
    def test_record_and_snapshot(self):
        mc = MetricsCollector()
        mc.record_stage("retrieval", 50.0)
        mc.record_stage("retrieval", 100.0)
        mc.record_query()
        mc.record_ingestion(42)
        mc.record_tokens(500)

        snap = mc.snapshot()
        assert snap["total_queries"] == 1
        assert snap["total_ingestions"] == 1
        assert snap["total_chunks_ingested"] == 42
        assert snap["estimated_tokens_used"] == 500
        assert snap["stages"]["retrieval"]["total_calls"] == 2
        assert snap["stages"]["retrieval"]["avg_latency_ms"] == 75.0
        assert snap["stages"]["retrieval"]["min_latency_ms"] == 50.0
        assert snap["stages"]["retrieval"]["max_latency_ms"] == 100.0

    def test_reset(self):
        mc = MetricsCollector()
        mc.record_query()
        mc.record_stage("test", 10.0)
        mc.reset()

        snap = mc.snapshot()
        assert snap["total_queries"] == 0
        assert snap["stages"] == {}

    def test_empty_snapshot(self):
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["total_queries"] == 0
        assert snap["stages"] == {}


class TestTimer:
    def test_timer_records_latency(self):
        # Use the global collector but test via a fresh stage name
        with Timer("test_stage_unique") as t:
            total = sum(range(1000))  # noqa: F841

        assert t.elapsed_ms > 0


class TestEstimateTokens:
    def test_estimate(self):
        assert estimate_tokens("hello world") > 0
        # ~4 chars per token
        assert estimate_tokens("a" * 400) == 100

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # minimum 1

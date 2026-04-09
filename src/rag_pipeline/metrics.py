"""Observability: latency tracking, token estimation, and metrics collection."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock


@dataclass
class StageMetric:
    """Timing and count data for a single pipeline stage."""

    total_calls: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls else 0.0


class MetricsCollector:
    """Thread-safe metrics collector for pipeline observability."""

    def __init__(self):
        self._lock = Lock()
        self._stages: dict[str, StageMetric] = defaultdict(StageMetric)
        self._total_queries: int = 0
        self._total_ingestions: int = 0
        self._total_chunks_ingested: int = 0
        self._total_tokens_estimated: int = 0

    def record_stage(self, stage: str, latency_ms: float) -> None:
        """Record latency for a pipeline stage."""
        with self._lock:
            m = self._stages[stage]
            m.total_calls += 1
            m.total_latency_ms += latency_ms
            m.min_latency_ms = min(m.min_latency_ms, latency_ms)
            m.max_latency_ms = max(m.max_latency_ms, latency_ms)

    def record_query(self) -> None:
        with self._lock:
            self._total_queries += 1

    def record_ingestion(self, chunks: int) -> None:
        with self._lock:
            self._total_ingestions += 1
            self._total_chunks_ingested += chunks

    def record_tokens(self, count: int) -> None:
        with self._lock:
            self._total_tokens_estimated += count

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of all metrics."""
        with self._lock:
            stages = {}
            for name, m in self._stages.items():
                stages[name] = {
                    "total_calls": m.total_calls,
                    "avg_latency_ms": round(m.avg_latency_ms, 2),
                    "min_latency_ms": round(m.min_latency_ms, 2) if m.total_calls else None,
                    "max_latency_ms": round(m.max_latency_ms, 2) if m.total_calls else None,
                }
            return {
                "total_queries": self._total_queries,
                "total_ingestions": self._total_ingestions,
                "total_chunks_ingested": self._total_chunks_ingested,
                "estimated_tokens_used": self._total_tokens_estimated,
                "stages": stages,
            }

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self._total_queries = 0
            self._total_ingestions = 0
            self._total_chunks_ingested = 0
            self._total_tokens_estimated = 0


# Global singleton
collector = MetricsCollector()


class Timer:
    """Context manager that records stage latency to the global collector."""

    def __init__(self, stage: str):
        self.stage = stage
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        collector.record_stage(self.stage, self.elapsed_ms)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)

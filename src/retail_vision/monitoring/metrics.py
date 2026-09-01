"""Per-camera health metrics and a light drift signal.

Operating 6,000 cameras is a monitoring problem more than a modelling problem.
Each worker keeps rolling stats that answer, per camera: is it alive, how fast,
how many people, how confident, how many identity decisions needed review, and
has the appearance distribution shifted (lighting change, camera moved).
Exposed as a dict for the summary endpoint and optionally as Prometheus gauges.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class DriftMonitor:
    """Compares the recent embedding mean against a reference mean (cosine distance).

    Cheap, model-agnostic and surprisingly effective at flagging a camera whose
    scene changed (new lighting, blocked lens, moved viewpoint).
    """

    def __init__(self, dim: int, reference_size: int = 200, window: int = 200) -> None:
        self.reference_size = reference_size
        self._reference: list[np.ndarray] = []
        self._ref_mean: np.ndarray | None = None
        self._recent: deque[np.ndarray] = deque(maxlen=window)
        self.dim = dim

    def observe(self, embedding: np.ndarray) -> None:
        if self._ref_mean is None:
            self._reference.append(embedding)
            if len(self._reference) >= self.reference_size:
                self._ref_mean = np.mean(self._reference, axis=0)
                self._reference.clear()
        else:
            self._recent.append(embedding)

    def score(self) -> float | None:
        """0 = identical distribution, 1 = orthogonal. None until the reference is built."""
        if self._ref_mean is None or len(self._recent) < 10:
            return None
        recent = np.mean(self._recent, axis=0)
        denom = np.linalg.norm(self._ref_mean) * np.linalg.norm(recent)
        if denom == 0:
            return None
        return float(1.0 - np.dot(self._ref_mean, recent) / denom)


class CameraMetrics:
    def __init__(self, camera_id: str, embedding_dim: int, window: int = 300) -> None:
        self.camera_id = camera_id
        self.started_at = time.time()
        self.frames = 0
        self.events = 0
        self.review_flags = 0
        self.identity_decisions = 0
        self._latency_ms: deque[float] = deque(maxlen=window)
        self._people: deque[int] = deque(maxlen=window)
        self._conf: deque[float] = deque(maxlen=window)
        self.drift = DriftMonitor(embedding_dim)
        self._prom = None

    def enable_prometheus(self) -> None:
        try:
            from prometheus_client import Gauge
        except ImportError:  # pragma: no cover - optional dependency
            return
        labels = ["camera_id"]
        self._prom = {
            "fps": Gauge("rva_camera_fps", "Processing FPS", labels),
            "latency": Gauge("rva_camera_latency_ms", "Per-frame latency p50 (ms)", labels),
            "people": Gauge("rva_camera_people", "People per frame (mean)", labels),
            "drift": Gauge("rva_camera_drift", "Embedding drift score", labels),
            "review": Gauge(
                "rva_camera_review_ratio", "Share of identity decisions needing review", labels
            ),
        }

    def record_frame(self, latency_s: float, n_people: int, confidences: list[float]) -> None:
        self.frames += 1
        self._latency_ms.append(latency_s * 1000.0)
        self._people.append(n_people)
        self._conf.extend(confidences)

    def record_identity(self, needs_review: bool, embedding: np.ndarray) -> None:
        self.identity_decisions += 1
        if needs_review:
            self.review_flags += 1
        self.drift.observe(embedding)

    def record_events(self, n: int) -> None:
        self.events += n

    def snapshot(self) -> dict:
        elapsed = max(time.time() - self.started_at, 1e-6)
        lat = np.array(self._latency_ms) if self._latency_ms else np.array([0.0])
        snap = {
            "camera_id": self.camera_id,
            "frames": self.frames,
            "fps": round(self.frames / elapsed, 2),
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "people_per_frame": round(float(np.mean(self._people)) if self._people else 0.0, 2),
            "mean_confidence": round(float(np.mean(self._conf)) if self._conf else 0.0, 3),
            "events": self.events,
            "identity_decisions": self.identity_decisions,
            "review_ratio": round(self.review_flags / max(self.identity_decisions, 1), 3),
            "drift_score": self.drift.score(),
        }
        if self._prom:
            lbl = {"camera_id": self.camera_id}
            self._prom["fps"].labels(**lbl).set(snap["fps"])
            self._prom["latency"].labels(**lbl).set(snap["latency_ms_p50"])
            self._prom["people"].labels(**lbl).set(snap["people_per_frame"])
            self._prom["review"].labels(**lbl).set(snap["review_ratio"])
            if snap["drift_score"] is not None:
                self._prom["drift"].labels(**lbl).set(snap["drift_score"])
        return snap

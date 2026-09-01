"""SORT-style multi-object tracker (Kalman prediction + Hungarian IoU matching).

Good enough for the POC and for the tests. In production the same interface
can be backed by ByteTrack (e.g. via `supervision`), which additionally
recovers low-confidence detections and is more robust to occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from retail_vision.config import TrackerConfig
from retail_vision.tracking.kalman import KalmanBoxFilter
from retail_vision.types import BBox, Detection, Track


@dataclass
class _TrackState:
    track_id: int
    kf: KalmanBoxFilter
    confidence: float
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    last_detection: np.ndarray | None = None


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ax1, ay1, ax2, ay2 = [a[:, i][:, None] for i in range(4)]
    bx1, by1, bx2, by2 = [b[:, i][None, :] for i in range(4)]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


class SortTracker:
    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self._tracks: list[_TrackState] = []
        self._next_id = 1
        self.frame_count = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame and return the confirmed tracks visible in this frame."""
        self.frame_count += 1

        predicted = np.array([t.kf.predict() for t in self._tracks]).reshape(-1, 4)
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1

        det_boxes = np.array([d.bbox.as_array() for d in detections]).reshape(-1, 4)
        matches, unmatched_dets, unmatched_trks = self._associate(det_boxes, predicted)

        for d_idx, t_idx in matches:
            t = self._tracks[t_idx]
            t.kf.update(det_boxes[d_idx])
            t.last_detection = det_boxes[d_idx]
            t.confidence = detections[d_idx].confidence
            t.hits += 1
            t.hit_streak += 1
            t.time_since_update = 0
            t.attributes.update(detections[d_idx].attributes)

        for t_idx in unmatched_trks:
            self._tracks[t_idx].hit_streak = 0

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            self._tracks.append(
                _TrackState(
                    track_id=self._next_id,
                    kf=KalmanBoxFilter(det_boxes[d_idx]),
                    confidence=det.confidence,
                    hits=1,
                    hit_streak=1,
                    attributes=dict(det.attributes),
                    last_detection=det_boxes[d_idx],
                )
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.cfg.max_age]

        out: list[Track] = []
        for t in self._tracks:
            confirmed = t.hits >= self.cfg.min_hits or self.frame_count <= self.cfg.min_hits
            if t.time_since_update == 0 and confirmed:
                # Report the raw matched detection: crops for ReID must not be smoothed.
                box = t.last_detection if t.last_detection is not None else t.kf.current()
                out.append(
                    Track(
                        track_id=t.track_id,
                        bbox=BBox(*map(float, box)),
                        confidence=t.confidence,
                        age=t.age,
                        hits=t.hits,
                        time_since_update=t.time_since_update,
                        attributes=dict(t.attributes),
                    )
                )
        return out

    def _associate(
        self, dets: np.ndarray, trks: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(trks) == 0 or len(dets) == 0:
            return [], list(range(len(dets))), list(range(len(trks)))
        iou = iou_matrix(dets, trks)
        rows, cols = linear_sum_assignment(-iou)
        matches: list[tuple[int, int]] = []
        for r, c in zip(rows, cols, strict=False):
            if iou[r, c] >= self.cfg.iou_threshold:
                matches.append((int(r), int(c)))
        matched_d = {m[0] for m in matches}
        matched_t = {m[1] for m in matches}
        unmatched_d = [i for i in range(len(dets)) if i not in matched_d]
        unmatched_t = [i for i in range(len(trks)) if i not in matched_t]
        return matches, unmatched_d, unmatched_t

"""Pixel-based mock detector (background subtraction + connected components).

Given the empty scene as reference it recovers the synthetic people with plain
OpenCV, so the whole pipeline (tracking, ReID, events, cloud ingest) can be
exercised end to end on any CPU with zero model downloads. It is the pixel-only
alternative to the OracleDetector and is deliberately weak under occlusion
(two overlapping people become one blob), which is useful for robustness tests.
"""

from __future__ import annotations

import cv2
import numpy as np

from retail_vision.detection.base import Detector
from retail_vision.types import BBox, Detection


class ColorBlobDetector(Detector):
    def __init__(
        self,
        min_area: int = 300,
        background_tolerance: int = 25,
        confidence: float = 0.9,
        dropout_rate: float = 0.0,
        seed: int | None = None,
        background: np.ndarray | None = None,
    ) -> None:
        self.background = background
        self.min_area = min_area
        self.background_tolerance = background_tolerance
        self.confidence = confidence
        self.dropout_rate = dropout_rate
        self._rng = np.random.default_rng(seed)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.background is not None:
            # Classic background subtraction against the empty scene.
            diff = np.abs(frame.astype(np.int16) - self.background.astype(np.int16)).sum(axis=2)
        else:
            # No reference: estimate a flat background from the frame border.
            border = np.concatenate([frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]])
            bg = np.median(border, axis=0)
            diff = np.abs(frame.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
        mask = (diff > self.background_tolerance).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            if self.dropout_rate and self._rng.random() < self.dropout_rate:
                continue  # simulate a missed detection
            patch = frame[y : y + h, x : x + w].reshape(-1, 3)
            dominant = np.median(patch, axis=0).astype(int).tolist()
            detections.append(
                Detection(
                    bbox=BBox(float(x), float(y), float(x + w), float(y + h)),
                    confidence=self.confidence,
                    attributes={"dominant_bgr": dominant},
                )
            )
        return detections

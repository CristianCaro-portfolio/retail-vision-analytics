"""Oracle detector: simulator ground truth with realistic detector noise.

Isolates the tracking / ReID / event logic from detector quality, which is how
those stages should be tested. Noise knobs mimic a real YOLO on CCTV:
box jitter, random misses, confidence spread, and dropping people who are
mostly occluded by someone closer to the camera.
"""

from __future__ import annotations

import numpy as np

from retail_vision.detection.base import Detector
from retail_vision.simulation.world import StoreWorld
from retail_vision.types import BBox, Detection


class OracleDetector(Detector):
    def __init__(
        self,
        world: StoreWorld,
        camera_id: str,
        jitter_px: float = 2.0,
        dropout_rate: float = 0.03,
        max_occlusion: float = 0.7,
        seed: int = 0,
    ) -> None:
        self.world = world
        self.camera_id = camera_id
        self.jitter_px = jitter_px
        self.dropout_rate = dropout_rate
        self.max_occlusion = max_occlusion
        self._rng = np.random.default_rng(seed)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        gt = self.world.ground_truth(self.camera_id)
        boxes = [BBox(*g["bbox"]) for g in gt]
        out: list[Detection] = []
        for i, box in enumerate(boxes):
            # Occlusion: fraction of this box covered by boxes with a lower foot point (closer).
            occluded = 0.0
            for j, other in enumerate(boxes):
                if j != i and other.y2 > box.y2:
                    ix = max(0.0, min(box.x2, other.x2) - max(box.x1, other.x1))
                    iy = max(0.0, min(box.y2, other.y2) - max(box.y1, other.y1))
                    occluded = max(occluded, ix * iy / max(box.area, 1e-6))
            if occluded > self.max_occlusion:
                continue
            if self._rng.random() < self.dropout_rate:
                continue
            j = self._rng.normal(0, self.jitter_px, size=4) if self.jitter_px else np.zeros(4)
            x1 = float(np.clip(box.x1 + j[0], 0, w))
            y1 = float(np.clip(box.y1 + j[1], 0, h))
            x2 = float(np.clip(box.x2 + j[2], 0, w))
            y2 = float(np.clip(box.y2 + j[3], 0, h))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            conf = float(np.clip(0.92 - 0.5 * occluded + self._rng.normal(0, 0.03), 0.3, 0.99))
            out.append(Detection(BBox(x1, y1, x2, y2), conf))
        return out

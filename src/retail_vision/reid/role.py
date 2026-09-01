"""Employee vs customer classification (first, binary level of identity).

In production this is a small classifier fine-tuned on crops (uniform, badge,
apron) or the detector itself trained with an extra `employee` class. For the
simulator we use the uniform colour hint the detector attaches to each blob.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from retail_vision.types import PersonRole, Track


class RoleClassifier(ABC):
    @abstractmethod
    def classify(self, track: Track, crop: np.ndarray) -> tuple[PersonRole, float]:
        """Return (role, confidence)."""


class UniformColorRoleClassifier(RoleClassifier):
    def __init__(self, uniform_bgr: tuple[int, int, int], tolerance: int = 40) -> None:
        self.uniform = np.array(uniform_bgr, dtype=int)
        self.tolerance = tolerance

    def classify(self, track: Track, crop: np.ndarray) -> tuple[PersonRole, float]:
        if crop.size == 0:
            return PersonRole.UNKNOWN, 0.0
        # The uniform is worn on the torso: sample the central torso band only (rows 20-50%,
        # middle 60% of columns) so head, floor and background pixels do not vote.
        h, w = crop.shape[:2]
        torso = crop[
            int(h * 0.2) : max(int(h * 0.2) + 1, int(h * 0.5)),
            int(w * 0.2) : max(int(w * 0.2) + 1, int(w * 0.8)),
        ]
        dominant = np.median(torso.reshape(-1, 3), axis=0)
        dist = float(np.abs(np.array(dominant, dtype=int) - self.uniform).sum())
        if dist <= self.tolerance:
            return PersonRole.EMPLOYEE, 1.0 - dist / (3 * 255)
        return PersonRole.CUSTOMER, min(1.0, dist / (3 * 255) + 0.5)

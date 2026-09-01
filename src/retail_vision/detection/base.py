from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from retail_vision.types import Detection


class Detector(ABC):
    """Anything that turns a BGR frame into a list of person detections."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: run a dummy inference so the first real frame is not slow."""
        return None

"""Frame sources: synthetic world viewports, video files and RTSP streams."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

import cv2
import numpy as np

from retail_vision.simulation.world import StoreWorld


class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp_seconds, bgr_frame)."""


class SyntheticSource(FrameSource):
    """Viewport into a shared StoreWorld. The world is advanced by the owner (see
    `SimulationRunner`), so several cameras render the same instant."""

    def __init__(self, world: StoreWorld, camera_id: str, fps: float = 10.0) -> None:
        self.world = world
        self.camera_id = camera_id
        self.fps = fps

    def current(self) -> tuple[float, np.ndarray]:
        return self.world.frame_idx / self.fps, self.world.render(self.camera_id)

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        while True:
            yield self.current()
            self.world.step()


class VideoSource(FrameSource):
    """OpenCV capture over a file path or an RTSP URL, with frame skipping to a target FPS."""

    def __init__(self, source: str, target_fps: float | None = None) -> None:
        self.source = source
        self.target_fps = target_fps

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source {self.source!r}")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(native_fps / self.target_fps))) if self.target_fps else 1
        idx = 0
        t0 = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ts = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else time.time() - t0
                    yield ts, frame
                idx += 1
        finally:
            cap.release()

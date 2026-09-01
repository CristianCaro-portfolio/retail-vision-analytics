"""Wires a StoreConfig into workers and runs them.

- StoreRuntime: builds shared components (embedder, employee gallery, resolver,
  dedup, sink) plus one CameraWorker per camera.
- SimulationRunner: drives all cameras in lock-step over a StoreWorld. Real
  deployments run one worker per camera process/thread over a VideoSource with
  the same StoreRuntime; see `rva run-camera`.
"""

from __future__ import annotations

from pathlib import Path

from retail_vision.config import StoreConfig
from retail_vision.detection.factory import build_detector
from retail_vision.events.dedup import EventDeduplicator
from retail_vision.pipeline.worker import CameraWorker
from retail_vision.reid.embedder import build_embedder
from retail_vision.reid.gallery import Gallery
from retail_vision.reid.resolver import IdentityResolver
from retail_vision.reid.role import RoleClassifier, UniformColorRoleClassifier
from retail_vision.simulation.sources import SyntheticSource, VideoSource
from retail_vision.simulation.world import UNIFORM_BGR, StoreWorld
from retail_vision.sinks import EventSink, build_sink
from retail_vision.types import Event


class StoreRuntime:
    def __init__(
        self,
        store: StoreConfig,
        sink: EventSink | None = None,
        employee_gallery: Gallery | None = None,
        role_classifier: RoleClassifier | None = None,
    ) -> None:
        self.store = store
        self.embedder = build_embedder(store.reid)
        if employee_gallery is None and store.reid.gallery_path:
            if Path(store.reid.gallery_path).exists():
                employee_gallery = Gallery.load(store.reid.gallery_path)
        self.employee_gallery = employee_gallery or Gallery(self.embedder.dim)
        self.resolver = IdentityResolver(store.reid, self.embedder.dim, self.employee_gallery)
        self.dedup = EventDeduplicator(store.events.dedup_window_seconds)
        self.sink = sink or build_sink(store.sink)
        self.role_classifier = role_classifier or UniformColorRoleClassifier(UNIFORM_BGR)
        self.workers: dict[str, CameraWorker] = {
            cam.camera_id: CameraWorker(
                store=store,
                camera=cam,
                detector=build_detector(store.detector),
                embedder=self.embedder,
                role_classifier=self.role_classifier,
                resolver=self.resolver,
                dedup=self.dedup,
                sink=self.sink,
            )
            for cam in store.cameras
        }

    def metrics(self) -> dict[str, dict]:
        return {cid: w.metrics.snapshot() for cid, w in self.workers.items()}

    def close(self) -> None:
        self.sink.close()

    def run_camera(self, camera_id: str, max_frames: int | None = None) -> int:
        """Run a single camera over its configured video/RTSP source. Returns frames processed."""
        cam = self.store.camera(camera_id)
        worker = self.workers[camera_id]
        source = VideoSource(cam.source, target_fps=cam.fps)
        n = 0
        for ts, frame in source.frames():
            worker.process_frame(ts, frame)
            self.resolver.expire(ts)
            n += 1
            if max_frames and n >= max_frames:
                break
        self.sink.flush()
        return n


class SimulationRunner:
    def __init__(self, runtime: StoreRuntime, world: StoreWorld) -> None:
        self.runtime = runtime
        self.world = world
        self.sources = {
            cam.camera_id: SyntheticSource(world, cam.camera_id, cam.fps)
            for cam in runtime.store.cameras
        }

    def run(self, frames: int) -> list[Event]:
        all_events: list[Event] = []
        for _ in range(frames):
            ts = None
            for camera_id, src in self.sources.items():
                ts, frame = src.current()
                all_events.extend(self.runtime.workers[camera_id].process_frame(ts, frame))
            if ts is not None:
                self.runtime.resolver.expire(ts)
                self.runtime.dedup.prune(ts)
            self.world.step()
        self.runtime.sink.flush()
        return all_events

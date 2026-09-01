"""Per-camera edge worker: frame in, business events out.

    frame -> detector -> tracker -> role classifier -> embedder -> identity resolver
          -> zone event engine -> deduplicator -> sink

The worker owns everything that is camera-local (detector, tracker, zones,
metrics). The identity resolver and the deduplicator are store-level and are
shared between the workers of one store.
"""

from __future__ import annotations

import time
from collections import Counter, deque

import numpy as np

from retail_vision.config import CameraConfig, StoreConfig
from retail_vision.detection.base import Detector
from retail_vision.events.dedup import EventDeduplicator
from retail_vision.events.zones import ZoneEventEngine
from retail_vision.monitoring.metrics import CameraMetrics
from retail_vision.reid.embedder import Embedder
from retail_vision.reid.quality import is_embeddable
from retail_vision.reid.resolver import IdentityResolver
from retail_vision.reid.role import RoleClassifier
from retail_vision.sinks import EventSink
from retail_vision.tracking.sort import SortTracker
from retail_vision.types import BBox, Event, Identity, PersonRole


class CameraWorker:
    def __init__(
        self,
        store: StoreConfig,
        camera: CameraConfig,
        detector: Detector,
        embedder: Embedder,
        role_classifier: RoleClassifier,
        resolver: IdentityResolver,
        dedup: EventDeduplicator,
        sink: EventSink,
        embed_every: int = 3,
    ) -> None:
        self.store = store
        self.camera = camera
        self.detector = detector
        self.embedder = embedder
        self.role_classifier = role_classifier
        self.resolver = resolver
        self.dedup = dedup
        self.sink = sink
        self.embed_every = embed_every
        self.tracker = SortTracker(store.tracker)
        self.events = ZoneEventEngine(store.store_id, camera, store.events)
        self.metrics = CameraMetrics(camera.camera_id, embedder.dim)
        self._embedding_cache: dict[int, np.ndarray] = {}
        self._role_votes: dict[int, deque[PersonRole]] = {}
        self._seen_tracks: set[int] = set()
        self.frame_idx = 0
        # (bbox, identity) pairs from the last processed frame; used by evaluation tools.
        self.last_assignments: list[tuple[BBox, Identity]] = []

    def process_frame(self, timestamp: float, frame: np.ndarray) -> list[Event]:
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)
        emitted: list[Event] = []
        current_ids: set[int] = set()
        self.last_assignments = []

        for track in tracks:
            current_ids.add(track.track_id)
            crop = track.bbox.crop(frame)
            if crop.size == 0:
                continue

            # Only clean crops feed the embedder (quality gate), and only every N frames per
            # track, which keeps GPU/CPU cost flat as the crowd grows.
            fresh = track.track_id not in self._embedding_cache
            if (fresh or self.frame_idx % self.embed_every == 0) and is_embeddable(
                track.bbox, frame.shape[1], frame.shape[0]
            ):
                self._embedding_cache[track.track_id] = self.embedder.embed(crop)
                role_now, _ = self.role_classifier.classify(track, crop)
                self._role_votes.setdefault(track.track_id, deque(maxlen=5)).append(role_now)
            if track.track_id not in self._embedding_cache:
                continue  # truncated at the frame border and never seen whole yet: wait
            embedding = self._embedding_cache[track.track_id]
            # Majority vote over recent frames absorbs single-frame occlusion glitches.
            role = Counter(self._role_votes[track.track_id]).most_common(1)[0][0]

            identity = self.resolver.resolve(
                self.camera.camera_id, track.track_id, role, embedding, timestamp
            )
            self.metrics.record_identity(identity.needs_review, embedding)
            self.last_assignments.append((track.bbox, identity))

            if track.track_id not in self._seen_tracks:
                self._seen_tracks.add(track.track_id)
                emitted.append(self._person_seen(identity, timestamp, track.bbox))

            emitted.extend(self.events.update(identity, track.bbox.foot_point, timestamp))

        # Tracks that disappeared: release bindings so a re-appearance is re-verified.
        for gone in set(self._embedding_cache) - current_ids:
            self._embedding_cache.pop(gone, None)
            self._role_votes.pop(gone, None)
            self.resolver.release_track(self.camera.camera_id, gone)
        missing_after = (self.store.tracker.max_age + 1) / max(self.camera.fps, 1e-6)
        emitted.extend(self.events.flush_missing(timestamp, missing_after))

        accepted = [e for e in emitted if self.dedup.accept(e)]
        for e in accepted:
            self.sink.emit(e)

        self.metrics.record_frame(
            time.perf_counter() - t0, len(tracks), [t.confidence for t in tracks]
        )
        self.metrics.record_events(len(accepted))
        self.frame_idx += 1
        return accepted

    def _person_seen(self, identity, timestamp: float, bbox) -> Event:
        return Event(
            event_type="person_seen",
            store_id=self.store.store_id,
            camera_id=self.camera.camera_id,
            timestamp=timestamp,
            global_id=identity.global_id,
            role=identity.role.value,
            employee_id=identity.employee_id,
            confidence=identity.confidence,
            payload={
                "needs_review": identity.needs_review,
                "bbox": [round(v, 1) for v in bbox.as_array().tolist()],
            },
        )

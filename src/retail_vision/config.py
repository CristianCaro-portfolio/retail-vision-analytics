"""Typed configuration for a store deployment.

A single YAML describes the store, its cameras, the zones drawn on each camera
and the thresholds used by the identity and event logic. The same file drives
the simulator, the edge workers and the tests.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class ZoneConfig(BaseModel):
    name: str
    # Polygon in pixel coordinates for this camera: [[x, y], ...]
    polygon: list[list[float]] = Field(min_length=3)
    kind: str = "area"  # area | entrance | shelf | seat | pos

    @field_validator("polygon")
    @classmethod
    def _points_2d(cls, v: list[list[float]]) -> list[list[float]]:
        for p in v:
            if len(p) != 2:
                raise ValueError("each polygon point must be [x, y]")
        return v


class CameraConfig(BaseModel):
    camera_id: str
    source: str = "synthetic"  # rtsp://..., file path, or "synthetic"
    width: int = 640
    height: int = 480
    fps: float = 10.0
    zones: list[ZoneConfig] = Field(default_factory=list)


class DetectorConfig(BaseModel):
    backend: str = "mock"  # mock | yolo | onnx
    model_path: str | None = None
    confidence_threshold: float = 0.4
    input_size: int = 640
    device: str = "cpu"


class TrackerConfig(BaseModel):
    max_age: int = 15  # frames without detection before a track is dropped
    min_hits: int = 3  # frames before a track is considered confirmed
    iou_threshold: float = 0.3


class ReIDConfig(BaseModel):
    backend: str = "histogram"  # histogram | osnet | onnx
    model_path: str | None = None
    embedding_dim: int = 128
    # Cosine similarity thresholds. Anything between the two is "needs review".
    match_threshold: float = 0.80
    review_threshold: float = 0.65
    gallery_path: str | None = None
    # Seconds a global identity stays alive without being seen by any camera.
    identity_ttl_seconds: float = 30.0
    # Max entries kept per identity in the online gallery.
    max_embeddings_per_identity: int = 20


class EventConfig(BaseModel):
    # Frames a person must be inside a zone before "entered" fires.
    zone_enter_frames: int = 3
    # Frames a person must be outside before "exited" fires.
    zone_exit_frames: int = 5
    # Minimum dwell (seconds) to emit a dwell event on exit.
    min_dwell_seconds: float = 1.0
    # Consecutive frames near a shelf to count as a product interaction.
    interaction_frames: int = 8
    # Seconds during which the same (identity, event, zone) is not re-emitted.
    dedup_window_seconds: float = 10.0


class SinkConfig(BaseModel):
    kind: str = "jsonl"  # jsonl | stdout | http | memory
    path: str = "data/events.jsonl"
    url: str | None = None
    batch_size: int = 50


class StoreConfig(BaseModel):
    store_id: str
    cameras: list[CameraConfig]
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    sink: SinkConfig = Field(default_factory=SinkConfig)

    def camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        raise KeyError(f"camera {camera_id!r} not found in store {self.store_id!r}")


def load_store_config(path: str | Path) -> StoreConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return StoreConfig.model_validate(raw)

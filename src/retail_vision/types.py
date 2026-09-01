"""Core data types shared across the pipeline.

Everything that flows between stages is a small, serialisable dataclass so
stages can be swapped (mock detector vs YOLO, in-memory sink vs HTTP) without
touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PersonRole(str, Enum):
    EMPLOYEE = "employee"
    CUSTOMER = "customer"
    UNKNOWN = "unknown"


@dataclass
class BBox:
    """Axis-aligned box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box; the best proxy for where a person stands."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = int(max(0, self.x1)), int(max(0, self.y1))
        x2, y2 = int(min(w, self.x2)), int(min(h, self.y2))
        return frame[y1:y2, x1:x2]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_name: str = "person"
    # Optional hint produced by the detector (e.g. uniform colour, badge).
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """A detection that has been associated over time within one camera."""

    track_id: int
    bbox: BBox
    confidence: float
    age: int  # frames since the track was created
    hits: int  # frames with a matched detection
    time_since_update: int
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Identity:
    """Result of resolving a track to a global identity."""

    global_id: str  # stable id across cameras (person-0012 or employee id)
    role: PersonRole
    employee_id: str | None = None
    confidence: float = 0.0
    needs_review: bool = False


@dataclass
class Event:
    """Business-level event emitted by the edge worker and shipped to the cloud."""

    event_type: str
    store_id: str
    camera_id: str
    timestamp: float
    global_id: str
    role: str
    zone: str | None = None
    employee_id: str | None = None
    confidence: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""A tiny synthetic store used to exercise the full pipeline without footage.

People are rendered as two-tone rectangles (torso colour / legs colour) walking
along waypoints in a shared world coordinate system. Each camera is a viewport
into that world, so the same actor is seen by several cameras with overlap,
which is exactly the cross-camera situation the identity resolver must handle.

Employees wear the store uniform colour on the torso; each employee has a
distinct trouser colour so the histogram embedder can tell them apart, the way
a real ReID model would use body appearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

BACKGROUND_BGR = (92, 88, 84)  # store floor
TILE_BGR = (104, 100, 96)
SKIN_BGR = (150, 180, 215)
UNIFORM_BGR = (200, 60, 30)  # store blue polo


@dataclass
class Furniture:
    """Static fixture drawn under the actors (shelves, counters) in world coordinates."""

    name: str
    rect: tuple[int, int, int, int]  # x1, y1, x2, y2
    color: tuple[int, int, int]


@dataclass
class Actor:
    name: str
    is_employee: bool
    torso_bgr: tuple[int, int, int]
    legs_bgr: tuple[int, int, int]
    waypoints: list[tuple[float, float]]
    speed: float = 12.0  # world px per frame
    size: tuple[int, int] = (36, 90)  # w, h
    start_frame: int = 0
    loop: bool = False
    _pos: np.ndarray = field(default_factory=lambda: np.zeros(2), repr=False)
    _target_idx: int = field(default=1, repr=False)
    _done: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._pos = np.array(self.waypoints[0], dtype=float)

    @property
    def employee_id(self) -> str | None:
        return self.name if self.is_employee else None

    def step(self, frame_idx: int) -> None:
        if frame_idx < self.start_frame or self._done:
            return
        if self._target_idx >= len(self.waypoints):
            if self.loop:
                self._target_idx = 0
            else:
                self._done = True
                return
        target = np.array(self.waypoints[self._target_idx], dtype=float)
        delta = target - self._pos
        dist = float(np.linalg.norm(delta))
        if dist <= self.speed:
            self._pos = target
            self._target_idx += 1
        else:
            self._pos = self._pos + delta / dist * self.speed

    def visible(self, frame_idx: int) -> bool:
        return frame_idx >= self.start_frame and not self._done

    def world_bbox(self) -> tuple[float, float, float, float]:
        w, h = self.size
        cx, foot_y = self._pos
        return (cx - w / 2, foot_y - h, cx + w / 2, foot_y)


@dataclass
class CameraView:
    camera_id: str
    offset: tuple[int, int]  # top-left of the viewport in world coords
    size: tuple[int, int] = (640, 480)


class StoreWorld:
    def __init__(
        self,
        actors: list[Actor],
        cameras: list[CameraView],
        world_size: tuple[int, int] = (1280, 960),
        noise_std: float = 0.0,
        seed: int = 0,
        furniture: list[Furniture] | None = None,
    ) -> None:
        self.actors = actors
        self.cameras = {c.camera_id: c for c in cameras}
        self.furniture = furniture or []
        self.world_size = world_size
        self.noise_std = noise_std
        self.frame_idx = 0
        self._rng = np.random.default_rng(seed)

    def step(self) -> None:
        for actor in self.actors:
            actor.step(self.frame_idx)
        self.frame_idx += 1

    def render_background(self, camera_id: str) -> np.ndarray:
        """The empty scene as this camera sees it (floor tiles + fixtures, no people)."""
        cam = self.cameras[camera_id]
        w, h = cam.size
        ox, oy = cam.offset
        frame = np.full((h, w, 3), BACKGROUND_BGR, dtype=np.uint8)
        tile = 80
        for x in range(-(ox % tile), w, tile):
            cv2.line(frame, (x, 0), (x, h), TILE_BGR, 1)
        for y in range(-(oy % tile), h, tile):
            cv2.line(frame, (0, y), (w, y), TILE_BGR, 1)
        for f in self.furniture:
            x1, y1, x2, y2 = f.rect
            cv2.rectangle(frame, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), f.color, -1)
            cv2.rectangle(
                frame,
                (x1 - ox, y1 - oy),
                (x2 - ox, y2 - oy),
                tuple(max(0, c - 40) for c in f.color),
                2,
            )
        return frame

    def render(self, camera_id: str, include_actors: bool = True) -> np.ndarray:
        cam = self.cameras[camera_id]
        w, h = cam.size
        frame = self.render_background(camera_id)
        if not include_actors:
            return frame
        ox, oy = cam.offset
        # Draw farthest (smallest y) first so nearer actors occlude.
        for actor in sorted(self.actors, key=lambda a: a.world_bbox()[3]):
            if not actor.visible(self.frame_idx):
                continue
            x1, y1, x2, y2 = actor.world_bbox()
            x1, x2 = int(x1 - ox), int(x2 - ox)
            y1, y2 = int(y1 - oy), int(y2 - oy)
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                continue
            _draw_person(frame, x1, y1, x2, y2, actor.torso_bgr, actor.legs_bgr)
        if self.noise_std > 0:
            noise = self._rng.normal(0, self.noise_std, frame.shape)
            frame = np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)
        return frame

    def ground_truth(self, camera_id: str) -> list[dict]:
        """Visible actors in camera coordinates; used by tests and evaluation."""
        cam = self.cameras[camera_id]
        ox, oy = cam.offset
        out = []
        for actor in self.actors:
            if not actor.visible(self.frame_idx):
                continue
            x1, y1, x2, y2 = actor.world_bbox()
            box = (x1 - ox, y1 - oy, x2 - ox, y2 - oy)
            if box[2] <= 0 or box[3] <= 0 or box[0] >= cam.size[0] or box[1] >= cam.size[1]:
                continue
            out.append({"name": actor.name, "is_employee": actor.is_employee, "bbox": box})
        return out

    def render_enrolment_crop(self, actor: Actor) -> np.ndarray:
        """Frontal crop used to build the employee gallery (stands in for enrolment photos)."""
        w, h = actor.size
        crop = np.full((h, w, 3), BACKGROUND_BGR, dtype=np.uint8)
        _draw_person(crop, 0, 0, w, h, actor.torso_bgr, actor.legs_bgr)
        return crop


def _draw_person(img, x1: int, y1: int, x2: int, y2: int, torso, legs) -> None:
    """Stylised person: shadow, legs, torso, head. Proportions roughly 15/35/50."""
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) // 2
    head_r = max(3, int(w * 0.28))
    neck = y1 + 2 * head_r
    hip = y1 + int(h * 0.55)
    cv2.ellipse(img, (cx, y2 - 2), (int(w * 0.6), 4), 0, 0, 360, (60, 58, 56), -1)
    cv2.rectangle(img, (x1 + int(w * 0.15), hip), (x2 - int(w * 0.15), y2), legs, -1)
    cv2.rectangle(img, (x1, neck), (x2, hip), torso, -1)
    cv2.circle(img, (cx, y1 + head_r), head_r, SKIN_BGR, -1)


def build_demo_world(seed: int = 0) -> StoreWorld:
    """Two overlapping cameras, two employees and three customers with scripted paths.

    World is 1280x960. cam-01 covers the left half (entrance + shelf A), cam-02 the
    right half (shelf B + a seated POS station). They overlap by 160px so people
    walking from left to right are seen by both.
    """
    rng = np.random.default_rng(seed)

    def rand_color() -> tuple[int, int, int]:
        return tuple(int(v) for v in rng.integers(20, 200, size=3))

    def customer(name: str, waypoints, speed: float, start: int) -> Actor:
        return Actor(
            name=name,
            is_employee=False,
            torso_bgr=rand_color(),
            legs_bgr=rand_color(),
            waypoints=waypoints,
            speed=speed,
            start_frame=start,
        )

    # World is 1280x960. The entrance is the left edge (world x < 120, y 240-720, i.e.
    # cam-01's "entrance" zone). Shelf A ~ world (200-320, 280-380); shelf B ~ (880-1020, 300-410);
    # POS seat ~ (1040-1120, 440-560).
    actors = [
        Actor(
            name="emp-001",
            is_employee=True,
            torso_bgr=UNIFORM_BGR,
            legs_bgr=(30, 30, 30),
            waypoints=[
                (80, 400),
                (300, 400),
                (300, 700),
                (900, 700),
                (1100, 500),
                (1100, 500),
                (1100, 500),
                (1100, 500),
                (900, 650),
                (400, 650),
                (80, 400),
            ],
            speed=10,
            start_frame=0,
            loop=True,
        ),
        Actor(
            name="emp-002",
            is_employee=True,
            torso_bgr=UNIFORM_BGR,
            legs_bgr=(150, 150, 150),
            waypoints=[
                (1200, 300),
                (900, 300),
                (700, 600),
                (400, 600),
                (100, 600),
                (100, 600),
                (400, 450),
                (900, 450),
                (1200, 300),
            ],
            speed=8,
            start_frame=15,
            loop=True,
        ),
        customer(
            "cust-A",
            [(20, 300), (250, 300), (250, 320), (250, 320), (250, 340), (800, 400), (1250, 400)],
            9,
            5,
        ),
        customer(
            "cust-B",
            [(20, 800), (500, 800), (950, 350), (950, 350), (950, 360), (1250, 200)],
            11,
            30,
        ),
        customer("cust-C", [(1250, 900), (700, 500), (250, 330), (250, 330), (20, 200)], 10, 60),
        customer(
            "cust-D",
            [
                (20, 500),
                (260, 360),
                (260, 360),
                (260, 370),
                (600, 800),
                (950, 380),
                (950, 380),
                (20, 700),
            ],
            10,
            140,
        ),
        customer(
            "cust-E",
            [(20, 650), (500, 500), (940, 370), (940, 370), (940, 380), (1250, 700)],
            12,
            200,
        ),
        customer(
            "cust-F", [(20, 350), (240, 350), (240, 350), (240, 360), (700, 700), (20, 450)], 9, 260
        ),
        customer(
            "cust-G",
            [(1250, 500), (960, 360), (960, 360), (300, 340), (300, 340), (20, 300)],
            11,
            320,
        ),
        customer("cust-H", [(20, 550), (230, 330), (230, 330), (900, 800), (1250, 850)], 10, 380),
    ]
    cameras = [
        CameraView("cam-01", offset=(0, 240)),
        CameraView("cam-02", offset=(480, 120)),
    ]
    furniture = [
        Furniture("shelf-A", (200, 240, 320, 300), (70, 110, 150)),
        Furniture("shelf-B", (880, 280, 1020, 330), (70, 110, 150)),
        Furniture("gondola", (520, 780, 760, 830), (70, 110, 150)),
        Furniture("pos-desk", (1040, 400, 1120, 450), (60, 60, 120)),
        Furniture("entrance-mat", (0, 300, 110, 640), (110, 90, 70)),
    ]
    return StoreWorld(actors, cameras, seed=seed, furniture=furniture)

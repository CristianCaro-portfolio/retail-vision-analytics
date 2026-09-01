"""Zone geometry and the per-person state machine that turns positions into events.

Events produced (all carry store_id, camera_id, global_id, role, zone):

- zone_entered        person confirmed inside a zone for N frames
- zone_exited         person confirmed outside for M frames; payload.dwell_seconds
- product_interaction person stayed by a `shelf` zone for K consecutive frames
                      (touch heuristic: proximity + persistence, no hand model needed for POC)
- seat_occupied / seat_released  same as enter/exit, for `seat` zones (workforce use-case)

Hysteresis (separate enter/exit frame counts) avoids flapping when a foot point
sits on a zone edge, which is the main source of double counting in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from retail_vision.config import CameraConfig, EventConfig
from retail_vision.types import Event, Identity


@dataclass
class Zone:
    name: str
    kind: str
    polygon: np.ndarray  # (N, 2) float32

    def contains(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0


def build_zones(camera: CameraConfig) -> list[Zone]:
    return [
        Zone(z.name, z.kind, np.array(z.polygon, dtype=np.float32).reshape(-1, 1, 2))
        for z in camera.zones
    ]


@dataclass
class _ZoneState:
    inside: bool = False
    inside_frames: int = 0
    outside_frames: int = 0
    entered_at: float | None = None
    interaction_fired: bool = False


@dataclass
class _PersonState:
    last_seen: float
    identity: Identity
    zones: dict[str, _ZoneState] = field(default_factory=dict)


class ZoneEventEngine:
    """One engine per camera; identities are global so events are comparable across cameras."""

    def __init__(
        self, store_id: str, camera: CameraConfig, cfg: EventConfig, zones: list[Zone] | None = None
    ) -> None:
        self.store_id = store_id
        self.camera_id = camera.camera_id
        self.cfg = cfg
        self.zones = zones if zones is not None else build_zones(camera)
        self._people: dict[str, _PersonState] = {}

    def update(
        self, identity: Identity, foot_point: tuple[float, float], timestamp: float
    ) -> list[Event]:
        person = self._people.setdefault(identity.global_id, _PersonState(timestamp, identity))
        person.last_seen = timestamp
        person.identity = identity
        events: list[Event] = []
        for zone in self.zones:
            state = person.zones.setdefault(zone.name, _ZoneState())
            if zone.contains(foot_point):
                state.inside_frames += 1
                state.outside_frames = 0
                if not state.inside and state.inside_frames >= self.cfg.zone_enter_frames:
                    state.inside = True
                    state.entered_at = timestamp
                    state.interaction_fired = False
                    events.append(self._event(self._enter_name(zone), identity, zone, timestamp))
                if (
                    state.inside
                    and zone.kind == "shelf"
                    and not state.interaction_fired
                    and state.inside_frames >= self.cfg.interaction_frames
                ):
                    state.interaction_fired = True
                    events.append(
                        self._event(
                            "product_interaction",
                            identity,
                            zone,
                            timestamp,
                            {"frames": state.inside_frames},
                        )
                    )
            else:
                state.outside_frames += 1
                state.inside_frames = 0
                if state.inside and state.outside_frames >= self.cfg.zone_exit_frames:
                    events.extend(self._exit(identity, zone, state, timestamp))
        return events

    def flush_missing(self, timestamp: float, missing_after: float) -> list[Event]:
        """Close open zone visits for people that disappeared (left the camera)."""
        events: list[Event] = []
        gone = [gid for gid, p in self._people.items() if timestamp - p.last_seen > missing_after]
        for gid in gone:
            person = self._people.pop(gid)
            for zone in self.zones:
                state = person.zones.get(zone.name)
                if state and state.inside:
                    events.extend(self._exit(person.identity, zone, state, timestamp))
        return events

    # Internals -------------------------------------------------------------------------

    def _exit(
        self, identity: Identity, zone: Zone, state: _ZoneState, timestamp: float
    ) -> list[Event]:
        dwell = timestamp - (state.entered_at or timestamp)
        state.inside = False
        state.entered_at = None
        payload = {"dwell_seconds": round(dwell, 3)}
        out = [self._event(self._exit_name(zone), identity, zone, timestamp, payload)]
        if dwell >= self.cfg.min_dwell_seconds:
            out.append(self._event("dwell", identity, zone, timestamp, payload))
        return out

    @staticmethod
    def _enter_name(zone: Zone) -> str:
        return "seat_occupied" if zone.kind == "seat" else "zone_entered"

    @staticmethod
    def _exit_name(zone: Zone) -> str:
        return "seat_released" if zone.kind == "seat" else "zone_exited"

    def _event(
        self,
        event_type: str,
        identity: Identity,
        zone: Zone,
        timestamp: float,
        payload: dict | None = None,
    ) -> Event:
        return Event(
            event_type=event_type,
            store_id=self.store_id,
            camera_id=self.camera_id,
            timestamp=timestamp,
            global_id=identity.global_id,
            role=identity.role.value,
            zone=zone.name,
            employee_id=identity.employee_id,
            confidence=identity.confidence,
            payload={
                "zone_kind": zone.kind,
                "needs_review": identity.needs_review,
                **(payload or {}),
            },
        )

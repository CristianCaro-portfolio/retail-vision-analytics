"""Cross-camera identity resolution.

Two-level identity, as the business needs it:

1. role  - employee vs customer (binary, cheap, high recall).
2. who   - employees are matched against an *authorised* enrolment gallery
           (consented photos per employee_id). Customers are matched only
           against an anonymous online gallery of people seen recently in the
           store, so the same shopper walking from camera 3 to camera 7 keeps
           one anonymous id and is not counted twice.

Every decision carries a confidence and an explicit "needs_review" flag. When
the similarity falls between the review and match thresholds the system does
not force an identity: the event is emitted as unknown/needs review and the
crop can be routed to a labelling queue to grow the gallery (hard negatives).

The resolver is per store and shared by all camera workers of that store. It is
deliberately in-memory here; a Redis-backed implementation with the same
interface is the natural next step when workers run on different edge boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retail_vision.config import ReIDConfig
from retail_vision.reid.gallery import Gallery
from retail_vision.types import Identity, PersonRole


@dataclass
class _ActiveIdentity:
    global_id: str
    role: PersonRole
    employee_id: str | None
    last_seen: float
    cameras: set[str] = field(default_factory=set)


class IdentityResolver:
    def __init__(
        self,
        cfg: ReIDConfig,
        embedding_dim: int,
        employee_gallery: Gallery | None = None,
        reverify_every: int = 30,
    ) -> None:
        self.cfg = cfg
        self.employee_gallery = employee_gallery or Gallery(embedding_dim)
        self.online_gallery = Gallery(embedding_dim, cfg.max_embeddings_per_identity)
        self.reverify_every = reverify_every
        self._active: dict[str, _ActiveIdentity] = {}
        self._track_binding: dict[tuple[str, str], tuple[str, int]] = {}  # -> (gid, frames)
        self._next_anonymous = 1
        self._next_unknown_employee = 1

    # Public API ------------------------------------------------------------------------

    def resolve(
        self,
        camera_id: str,
        track_id: int,
        role: PersonRole,
        embedding: np.ndarray,
        timestamp: float,
    ) -> Identity:
        key = (camera_id, str(track_id))
        bound = self._track_binding.get(key)
        if bound is not None:
            gid, frames = bound
            frames += 1
            active = self._active.get(gid)
            if active is not None and frames % self.reverify_every != 0:
                self._touch(active, camera_id, timestamp, embedding)
                self._track_binding[key] = (gid, frames)
                return Identity(gid, active.role, active.employee_id, confidence=1.0)
            # binding expired or periodic re-verification: fall through and re-resolve

        identity = (
            self._resolve_employee(embedding)
            if role == PersonRole.EMPLOYEE
            else self._resolve_customer(embedding)
        )
        if bound is not None:
            # Re-verification is only allowed to *improve* a binding: a confident
            # employee match may replace an unknown one, but a momentary occlusion
            # must not turn a known employee into a new anonymous identity.
            prev = self._active.get(bound[0])
            if prev is not None and (identity.needs_review or identity.confidence <= 0.0):
                self._touch(prev, camera_id, timestamp, embedding, learn=False)
                self._track_binding[key] = (prev.global_id, bound[1] + 1)
                return Identity(prev.global_id, prev.role, prev.employee_id, confidence=1.0)
        active = self._active.get(identity.global_id)
        if active is None:
            active = _ActiveIdentity(
                identity.global_id, identity.role, identity.employee_id, timestamp
            )
            self._active[identity.global_id] = active
        self._touch(active, camera_id, timestamp, embedding)
        self._track_binding[key] = (identity.global_id, 1)
        return identity

    def release_track(self, camera_id: str, track_id: int) -> None:
        self._track_binding.pop((camera_id, str(track_id)), None)

    def expire(self, now: float) -> list[str]:
        """Drop identities not seen by any camera for longer than the TTL."""
        expired = [
            gid
            for gid, a in self._active.items()
            if now - a.last_seen > self.cfg.identity_ttl_seconds
        ]
        for gid in expired:
            self._active.pop(gid, None)
            self.online_gallery.remove(gid)
        if expired:
            gone = set(expired)
            self._track_binding = {k: v for k, v in self._track_binding.items() if v[0] not in gone}
        return expired

    def active_identities(self) -> dict[str, dict]:
        return {
            gid: {
                "role": a.role.value,
                "employee_id": a.employee_id,
                "last_seen": a.last_seen,
                "cameras": sorted(a.cameras),
            }
            for gid, a in self._active.items()
        }

    # Internals -------------------------------------------------------------------------

    def _touch(
        self,
        active: _ActiveIdentity,
        camera_id: str,
        ts: float,
        embedding: np.ndarray,
        learn: bool = True,
    ) -> None:
        active.last_seen = ts
        active.cameras.add(camera_id)
        if not learn:
            return
        # Online learning guard: only embeddings that already resemble the identity are
        # added, so an occluded or merged crop cannot poison the gallery.
        if active.global_id in self.online_gallery:
            sim = dict(self.online_gallery.query(embedding, top_k=len(self.online_gallery))).get(
                active.global_id, 0.0
            )
            if sim < self.cfg.review_threshold:
                return
        self.online_gallery.add(active.global_id, embedding)

    def _resolve_employee(self, embedding: np.ndarray) -> Identity:
        candidates = self.employee_gallery.query(embedding, top_k=2)
        if candidates:
            best_id, best_sim = candidates[0]
            # Margin check: reject ambiguous matches between two look-alike employees.
            margin = best_sim - (candidates[1][1] if len(candidates) > 1 else -1.0)
            if best_sim >= self.cfg.match_threshold and margin > 0.02:
                return Identity(best_id, PersonRole.EMPLOYEE, best_id, best_sim, False)
            if best_sim >= self.cfg.review_threshold:
                return Identity(best_id, PersonRole.EMPLOYEE, best_id, best_sim, True)
        # Unknown employee: keep a stable anonymous id so we can still track them.
        online = self._query_online(embedding, PersonRole.EMPLOYEE)
        if online is not None:
            gid, sim = online
            return Identity(gid, PersonRole.EMPLOYEE, None, sim, True)
        gid = f"employee-unknown-{self._next_unknown_employee:04d}"
        self._next_unknown_employee += 1
        return Identity(gid, PersonRole.EMPLOYEE, None, 0.0, True)

    def _resolve_customer(self, embedding: np.ndarray) -> Identity:
        online = self._query_online(embedding, PersonRole.CUSTOMER)
        if online is not None:
            gid, sim = online
            return Identity(gid, PersonRole.CUSTOMER, None, sim, False)
        gid = f"person-{self._next_anonymous:04d}"
        self._next_anonymous += 1
        return Identity(gid, PersonRole.CUSTOMER, None, 0.0, False)

    def _query_online(self, embedding: np.ndarray, role: PersonRole) -> tuple[str, float] | None:
        for gid, sim in self.online_gallery.query(embedding, top_k=5):
            active = self._active.get(gid)
            if active is None or active.role != role:
                continue
            if sim >= self.cfg.match_threshold:
                return gid, sim
        return None

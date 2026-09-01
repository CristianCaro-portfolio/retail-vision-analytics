"""Embedding gallery: identity_id -> set of embeddings.

Used both for the authorised employee gallery (built offline from enrolment
photos, persisted to disk) and for the online customer gallery (built at
runtime, expires with the identity TTL).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np


class Gallery:
    def __init__(self, dim: int, max_per_identity: int = 20) -> None:
        self.dim = dim
        self.max_per_identity = max_per_identity
        self._store: dict[str, deque[np.ndarray]] = {}
        self._meta: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, identity_id: str) -> bool:
        return identity_id in self._store

    def ids(self) -> list[str]:
        return list(self._store.keys())

    def add(self, identity_id: str, embedding: np.ndarray, meta: dict | None = None) -> None:
        if embedding.shape != (self.dim,):
            raise ValueError(f"embedding must be ({self.dim},), got {embedding.shape}")
        bucket = self._store.setdefault(identity_id, deque(maxlen=self.max_per_identity))
        bucket.append(embedding.astype(np.float32))
        if meta:
            self._meta.setdefault(identity_id, {}).update(meta)

    def remove(self, identity_id: str) -> None:
        self._store.pop(identity_id, None)
        self._meta.pop(identity_id, None)

    def meta(self, identity_id: str) -> dict:
        return self._meta.get(identity_id, {})

    def query(self, embedding: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """Return (identity_id, similarity) sorted desc. Similarity = max over the bucket.

        Using the max (instead of the centroid) is more robust when a gallery
        mixes several viewpoints of the same person.
        """
        if not self._store:
            return []
        ids: list[str] = []
        sims: list[float] = []
        for identity_id, bucket in self._store.items():
            mat = np.stack(bucket)
            ids.append(identity_id)
            sims.append(float((mat @ embedding).max()))
        order = np.argsort(sims)[::-1][:top_k]
        return [(ids[i], sims[i]) for i in order]

    # Persistence -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self._store.items()}
        np.savez_compressed(path, **arrays)
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"dim": self.dim, "max_per_identity": self.max_per_identity, "meta": self._meta},
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> Gallery:
        path = Path(path)
        with open(path.with_suffix(".meta.json"), encoding="utf-8") as fh:
            info = json.load(fh)
        gal = cls(dim=info["dim"], max_per_identity=info["max_per_identity"])
        gal._meta = info.get("meta", {})
        with np.load(path) as data:
            for identity_id in data.files:
                for row in data[identity_id]:
                    gal.add(identity_id, row)
        return gal

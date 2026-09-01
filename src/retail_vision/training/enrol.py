"""Employee enrolment: build the authorised gallery from consented photos.

Expected layout:  <root>/<employee_id>/*.jpg   (full-body crops, several angles)

Enrolment is the only place where identity photos are used; the edge workers
receive embeddings only. Keep the photo store under the client's access
controls and re-run enrolment when uniforms or staff change.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from retail_vision.reid.embedder import Embedder
from retail_vision.reid.gallery import Gallery
from retail_vision.training.augment import IMAGE_EXTS


def build_employee_gallery(
    root: str | Path, embedder: Embedder, max_per_identity: int = 20
) -> tuple[Gallery, dict]:
    root = Path(root)
    gallery = Gallery(embedder.dim, max_per_identity)
    stats: dict[str, int] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        n = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gallery.add(person_dir.name, embedder.embed(img), {"source": str(person_dir)})
            n += 1
        stats[person_dir.name] = n
    return gallery, stats


def gallery_arrays(gallery: Gallery) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a gallery to (embeddings, ids) for evaluation helpers."""
    embs, ids = [], []
    for identity in gallery.ids():
        for row in gallery._store[identity]:
            embs.append(row)
            ids.append(identity)
    return np.stack(embs), np.array(ids)

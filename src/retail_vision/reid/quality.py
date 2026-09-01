"""Crop quality gate.

Feeding truncated or tiny crops to the embedder is the fastest way to pollute a
gallery: a person half outside the frame produces an embedding that matches
nobody, so the resolver mints a new identity and the store double counts.
"""

from __future__ import annotations

from retail_vision.types import BBox


def is_embeddable(
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    min_height: int = 32,
    border_margin: int = 2,
    max_aspect: float = 1.2,
) -> bool:
    if bbox.height < min_height:
        return False
    if (
        bbox.x1 <= border_margin
        or bbox.y1 <= border_margin
        or bbox.x2 >= frame_w - border_margin
        or bbox.y2 >= frame_h - border_margin
    ):
        return False
    # A standing person is taller than wide; very wide boxes are usually merged people.
    return bbox.width / max(bbox.height, 1e-6) <= max_aspect

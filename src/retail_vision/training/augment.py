"""Bounding-box aware augmentation for small retail datasets.

A new store often ships with a handful of reference images per camera angle.
This module expands a YOLO-format dataset by a configurable factor with
augmentations chosen for CCTV footage: viewpoint jitter (crop/affine/flip),
illumination changes (brightness, contrast, gamma, shadows), sensor artefacts
(blur, noise, JPEG) and partial occlusion (coarse dropout). Boxes are
transformed with the image and dropped when mostly cut out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class YoloSample:
    image_path: Path
    label_path: Path

    def read(self) -> tuple[np.ndarray, list[list[float]], list[int]]:
        img = cv2.imread(str(self.image_path))
        boxes: list[list[float]] = []
        classes: list[int] = []
        if self.label_path.exists():
            for line in self.label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                classes.append(int(parts[0]))
                boxes.append([float(v) for v in parts[1:]])
        return img, boxes, classes


def build_cctv_pipeline(image_size: int = 640, strength: str = "medium"):
    """Albumentations Compose tuned for fixed retail cameras."""
    import albumentations as A  # noqa: N812

    s = {"light": 0.5, "medium": 1.0, "heavy": 1.5}[strength]
    return A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(height=image_size, width=image_size, erosion_rate=0.1, p=0.6),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(1 - 0.15 * s, 1 + 0.15 * s),
                translate_percent=(-0.05 * s, 0.05 * s),
                rotate=(-6 * s, 6 * s),
                shear=(-4 * s, 4 * s),
                p=0.7,
            ),
            A.Perspective(scale=(0.02, 0.05 * s), p=0.3),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(0.3 * s, 0.3 * s, p=1.0),
                    A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                    A.ColorJitter(0.3 * s, 0.3 * s, 0.3 * s, 0.05 * s, p=1.0),
                ],
                p=0.9,
            ),
            A.RandomShadow(p=0.2),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Defocus(radius=(1, 3), p=1.0),
                ],
                p=0.3,
            ),
            A.GaussNoise(p=0.3),
            A.ImageCompression(quality_range=(40, 90), p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.2),
                hole_width_range=(0.05, 0.2),
                p=0.3 * s,
            ),
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, fill=114),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3, min_area=64
        ),
    )


def list_samples(images_dir: Path, labels_dir: Path) -> list[YoloSample]:
    out = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() in IMAGE_EXTS:
            out.append(YoloSample(img, labels_dir / f"{img.stem}.txt"))
    return out


def expand_dataset(
    src_images: str | Path,
    src_labels: str | Path,
    dst_dir: str | Path,
    factor: int = 20,
    image_size: int = 640,
    strength: str = "medium",
    seed: int = 0,
    keep_originals: bool = True,
) -> dict:
    """Write `factor` augmented copies of every sample into dst_dir/{images,labels}."""
    random.seed(seed)
    np.random.seed(seed)
    src_images, src_labels, dst_dir = Path(src_images), Path(src_labels), Path(dst_dir)
    out_img, out_lbl = dst_dir / "images", dst_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    pipeline = build_cctv_pipeline(image_size, strength)
    samples = list_samples(src_images, src_labels)
    written = dropped_boxes = 0
    for sample in samples:
        img, boxes, classes = sample.read()
        if img is None:
            continue
        if keep_originals:
            cv2.imwrite(str(out_img / sample.image_path.name), img)
            _write_labels(out_lbl / f"{sample.image_path.stem}.txt", boxes, classes)
            written += 1
        for k in range(factor):
            aug = pipeline(image=img, bboxes=boxes, class_labels=classes)
            dropped_boxes += len(boxes) - len(aug["bboxes"])
            name = f"{sample.image_path.stem}_aug{k:03d}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), aug["image"])
            _write_labels(out_lbl / f"{name}.txt", aug["bboxes"], aug["class_labels"])
            written += 1
    return {
        "source_images": len(samples),
        "written_images": written,
        "dropped_boxes": dropped_boxes,
        "output_dir": str(dst_dir),
    }


def _write_labels(path: Path, boxes, classes) -> None:
    lines = [
        f"{int(c)} " + " ".join(f"{float(v):.6f}" for v in b)
        for b, c in zip(boxes, classes, strict=False)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))

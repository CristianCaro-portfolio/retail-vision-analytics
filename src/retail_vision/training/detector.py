"""Detector fine-tuning and export (Ultralytics YOLO).

Typical loop for a new client:
  1. label ~100 frames per camera angle (person, employee) -> YOLO format
  2. `rva augment` to expand x20-x100
  3. `rva train-detector` starting from a COCO checkpoint
  4. export to ONNX / TensorRT for the edge boxes
"""

from __future__ import annotations

from pathlib import Path

import yaml


def write_data_yaml(
    root: str | Path, class_names: list[str], train: str = "train/images", val: str = "val/images"
) -> Path:
    root = Path(root)
    data = {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "names": dict(enumerate(class_names)),
    }
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def train_detector(
    data_yaml: str | Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "retail",
    freeze: int | None = 10,
) -> Path:
    """Fine-tune and return the path to best.pt. Freezing the backbone is the safe default
    for small datasets; unfreeze once you have thousands of real (not augmented) frames."""
    from ultralytics import YOLO

    model = YOLO(base_model)
    kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        # built-in augmentation on top of the offline expansion
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.5,
        fliplr=0.5,
        patience=15,
    )
    if freeze:
        kwargs["freeze"] = freeze
    model.train(**kwargs)
    return Path(project) / name / "weights" / "best.pt"


def export_detector(
    weights: str | Path, fmt: str = "onnx", imgsz: int = 640, half: bool = False
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = model.export(format=fmt, imgsz=imgsz, half=half, simplify=True, dynamic=False)
    return Path(out)


def evaluate_detector(weights: str | Path, data_yaml: str | Path, imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=str(data_yaml), imgsz=imgsz, verbose=False)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

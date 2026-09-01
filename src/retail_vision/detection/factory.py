from __future__ import annotations

from retail_vision.config import DetectorConfig
from retail_vision.detection.base import Detector
from retail_vision.detection.mock import ColorBlobDetector


def build_detector(cfg: DetectorConfig) -> Detector:
    if cfg.backend == "mock":
        return ColorBlobDetector(confidence=0.9)
    if cfg.backend == "yolo":
        from retail_vision.detection.yolo import YoloDetector

        return YoloDetector(
            model_path=cfg.model_path or "yolov8n.pt",
            confidence_threshold=cfg.confidence_threshold,
            input_size=cfg.input_size,
            device=cfg.device,
        )
    if cfg.backend == "onnx":
        from retail_vision.detection.yolo import OnnxDetector

        if not cfg.model_path:
            raise ValueError("detector.model_path is required for the onnx backend")
        return OnnxDetector(
            model_path=cfg.model_path,
            confidence_threshold=cfg.confidence_threshold,
            input_size=cfg.input_size,
        )
    raise ValueError(f"unknown detector backend {cfg.backend!r}")

"""Real detectors: Ultralytics YOLO (PyTorch) and a generic ONNX Runtime backend.

Both are optional dependencies; the pipeline imports them lazily so the edge
image can stay small when only the mock detector is needed (tests, CI).
"""

from __future__ import annotations

import cv2
import numpy as np

from retail_vision.detection.base import Detector
from retail_vision.types import BBox, Detection

PERSON_CLASS_ID = 0  # COCO


class YoloDetector(Detector):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        input_size: int = 640,
        device: str = "cpu",
        classes: tuple[int, ...] = (PERSON_CLASS_ID,),
    ) -> None:
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.device = device
        self.classes = list(classes)

    def warmup(self) -> None:
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for xyxy, conf, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                strict=False,
            ):
                out.append(
                    Detection(
                        bbox=BBox(*map(float, xyxy)),
                        confidence=float(conf),
                        class_name=self.model.names.get(int(cls), str(int(cls))),
                    )
                )
        return out


class OnnxDetector(Detector):
    """Minimal ONNX Runtime wrapper for a YOLOv8-style export (1x84x8400 output).

    This is the backend to use on edge boxes without PyTorch: export once with
    `yolo export model=best.pt format=onnx` and ship only the .onnx file.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        input_size: int = 640,
        iou_threshold: float = 0.5,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort  # lazy import

        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), scale, (w, h)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, scale, (w, h) = self._preprocess(frame)
        pred = self.session.run(None, {self.input_name: blob})[0]  # (1, 84, N)
        pred = np.squeeze(pred, 0).T  # (N, 84)
        boxes_cxcywh, scores = pred[:, :4], pred[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        keep = (confs >= self.confidence_threshold) & (cls_ids == PERSON_CLASS_ID)
        boxes_cxcywh, confs = boxes_cxcywh[keep], confs[keep]
        if len(boxes_cxcywh) == 0:
            return []
        xyxy = np.empty_like(boxes_cxcywh)
        xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) / scale
        xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) / scale
        xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) / scale
        xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) / scale
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        idx = cv2.dnn.NMSBoxes(
            [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy],
            confs.tolist(),
            self.confidence_threshold,
            self.iou_threshold,
        )
        idx = np.array(idx).reshape(-1)
        return [Detection(bbox=BBox(*map(float, xyxy[i])), confidence=float(confs[i])) for i in idx]

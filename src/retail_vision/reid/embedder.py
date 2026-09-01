"""Appearance embedders for re-identification.

All embedders return L2-normalised vectors so cosine similarity is a dot product.

- HistogramEmbedder: colour-histogram baseline. No model needed; used by the
  simulator/tests and as a sanity baseline when evaluating a real model.
- OnnxEmbedder: any exported ReID backbone (OSNet, FastReID, ...) as ONNX.
- TorchreidEmbedder: OSNet via torchreid for training/eval on a GPU box.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from retail_vision.config import ReIDConfig


def l2_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, crop: np.ndarray) -> np.ndarray:
        """BGR crop (H, W, 3) -> (dim,) unit vector."""

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed(c) for c in crops])


class HistogramEmbedder(Embedder):
    """HSV histogram over a 2-part vertical split (torso / legs).

    Splitting the body keeps some spatial information, which matters in retail
    (an employee polo + dark trousers vs a customer with the same trouser colour).
    """

    def __init__(self, bins: tuple[int, int, int] = (8, 4, 4), parts: int = 2) -> None:
        self.bins = bins
        self.parts = parts
        self.dim = bins[0] * bins[1] * bins[2] * parts

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        # Light blur first: sensor noise must not move pixels across histogram bins.
        hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
        h = hsv.shape[0]
        chunks = []
        for p in range(self.parts):
            part = hsv[p * h // self.parts : (p + 1) * h // self.parts]
            hist = cv2.calcHist([part], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256])
            chunks.append(hist.flatten())
        vec = np.concatenate(chunks).astype(np.float32)
        return l2_normalize(vec)


class OnnxEmbedder(Embedder):
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (128, 256),  # (w, h), OSNet default
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, self.input_size)[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        return img.transpose(2, 0, 1)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop])[0]

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = np.stack([self._prep(c) for c in crops]).astype(np.float32)
        out = self.session.run(None, {self.input_name: batch})[0]
        return l2_normalize(out.astype(np.float32))


class TorchreidEmbedder(Embedder):
    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu") -> None:
        import torchreid  # lazy

        self.extractor = torchreid.utils.FeatureExtractor(model_name=model_name, device=device)
        self.dim = 512

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop])[0]

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        feats = self.extractor([c[:, :, ::-1] for c in crops]).cpu().numpy()
        return l2_normalize(feats.astype(np.float32))


def build_embedder(cfg: ReIDConfig) -> Embedder:
    if cfg.backend == "histogram":
        return HistogramEmbedder()
    if cfg.backend == "onnx":
        if not cfg.model_path:
            raise ValueError("reid.model_path is required for the onnx backend")
        return OnnxEmbedder(cfg.model_path)
    if cfg.backend == "osnet":
        return TorchreidEmbedder(model_name=cfg.model_path or "osnet_x0_25")
    raise ValueError(f"unknown reid backend {cfg.backend!r}")

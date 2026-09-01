"""Constant-velocity Kalman filter over (cx, cy, area, aspect_ratio).

Same state parameterisation as SORT. Kept dependency-free (numpy only) so the
edge worker does not need filterpy.
"""

from __future__ import annotations

import numpy as np


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / max(h, 1e-6)], dtype=float).reshape(4, 1)


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    cx, cy, s, r = x[0, 0], x[1, 0], max(x[2, 0], 1e-6), max(x[3, 0], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


class KalmanBoxFilter:
    def __init__(self, bbox: np.ndarray) -> None:
        self.dim_x, self.dim_z = 7, 4
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0  # high uncertainty on unobserved velocities
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)

    def predict(self) -> np.ndarray:
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_bbox(self.x)

    def update(self, bbox: np.ndarray) -> None:
        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(7) - k @ self.H) @ self.P

    def current(self) -> np.ndarray:
        return x_to_bbox(self.x)

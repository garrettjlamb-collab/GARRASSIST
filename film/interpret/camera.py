"""The camera behind a court mapping. A homography from the floor to the picture holds the camera: with square pixels and the
optical centre in the middle of the picture, the focal length and pose follow (Zhang's method), so a point above the floor can
be placed in the picture, which is what the ball's flight needs. Feet: x, y on the floor as in film/interpret/court.py, z up."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    P: np.ndarray                      # 3 x 4: homogeneous court feet -> homogeneous pixels
    focal: float
    centre: np.ndarray                 # where the camera is, in court feet (z is its height)

    def project(self, point) -> tuple[float, float] | None:
        v = self.P @ np.array([*point, 1.0])
        return None if v[2] <= 1e-9 else (float(v[0] / v[2]), float(v[1] / v[2]))

    def ray(self, px: float, py: float) -> np.ndarray:
        """The direction, in court feet, from the camera through a pixel."""
        d = np.linalg.solve(self.P[:, :3], np.array([px, py, 1.0]))
        return d / np.linalg.norm(d)


def from_homography(H: np.ndarray, width: int, height: int) -> Camera | None:
    """`H` is pixels -> feet, as Mapping.H. None when the numbers do not make a camera: the floor seen straight on, or from below."""
    back = np.linalg.inv(H)
    shift = np.array([[1.0, 0.0, -width / 2], [0.0, 1.0, -height / 2], [0.0, 0.0, 1.0]])
    a = shift @ back
    a1, a2 = a[:, 0], a[:, 1]
    # r1 . r2 = 0 and |r1| = |r2| for K = diag(f, f, 1); use whichever is better conditioned and average when both are valid.
    estimates = []
    if abs(a1[2] * a2[2]) > 1e-18:
        f2 = -(a1[0] * a2[0] + a1[1] * a2[1]) / (a1[2] * a2[2])
        if f2 > 0:
            estimates.append(f2)
    d = a1[2] ** 2 - a2[2] ** 2
    if abs(d) > 1e-18:
        f2 = -((a1[0] ** 2 + a1[1] ** 2) - (a2[0] ** 2 + a2[1] ** 2)) / d
        if f2 > 0:
            estimates.append(f2)
    if not estimates:
        return None
    f = float(np.sqrt(np.median(estimates)))
    K = np.array([[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]])
    b = np.linalg.solve(K, a)
    scale = 2.0 / (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1]))
    b = b * scale
    if b[2, 2] < 0:                    # the floor in front of the camera
        b = -b
    r1, r2, t = b[:, 0], b[:, 1], b[:, 2]
    r3 = np.cross(r1, r2)
    # Up in the court must be up in the picture.
    if r3[1] > 0:
        r3 = -r3
    Rt = np.column_stack([r1, r2, r3, t])
    P = np.linalg.inv(shift) @ K @ Rt
    centre = -np.linalg.solve(np.column_stack([r1, r2, r3]), t)
    return Camera(P=P, focal=f, centre=centre)

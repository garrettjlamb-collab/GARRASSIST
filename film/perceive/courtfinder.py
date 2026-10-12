"""Finds the court's named points in a frame and fits the pixels-to-feet mapping (film.interpret.court)."""
from __future__ import annotations

import numpy as np

from film.common import registry
from film.interpret.court import POINT_NAMES, Mapping, fit_mapping
from film.perceive.detectors import IMGSZ

MIN_KPT_CONF = 0.5


def _np(t):
    """Ultralytics hands back torch tensors on the device it ran on (mps on the laptop); numpy otherwise."""
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def points_from_result(result) -> dict[str, tuple[float, float]]:
    """Named points from an ultralytics pose result, the most confident court instance only."""
    kp = getattr(result, "keypoints", None)
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return {}
    best = int(np.argmax(_np(result.boxes.conf))) if result.boxes is not None and len(result.boxes.conf) else 0
    xy = _np(kp.xy[best])
    if len(xy) != len(POINT_NAMES):
        raise ValueError(f"court model gives {len(xy)} points, expected {len(POINT_NAMES)}")
    conf = _np(kp.conf[best]) if kp.conf is not None else np.ones(len(xy))
    return {name: (float(x), float(y)) for name, (x, y), c in zip(POINT_NAMES, xy, conf) if c >= MIN_KPT_CONF}


class CourtFinder:
    def __init__(self, model, device: str, imgsz: int, version: str = "court@test"):
        self.model, self.device, self.imgsz, self.version = model, device, imgsz, version

    @classmethod
    def load(cls, device: str) -> "CourtFinder | None":
        found = registry.latest("court")
        if found is None:
            return None
        version, folder = found
        from ultralytics import YOLO

        info = registry.read_manifest(folder) if (folder / "manifest.json").exists() else {}
        return cls(YOLO(str(folder / "best.pt")), device, int(info.get("imgsz") or IMGSZ["court"]), version=f"court@{version}")

    def points(self, frame_bgr: np.ndarray) -> dict[str, tuple[float, float]]:
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, device=self.device, verbose=False, conf=0.25)
        return points_from_result(results[0]) if results else {}

    def mapping(self, frame_bgr: np.ndarray) -> Mapping | None:
        return fit_mapping(self.points(frame_bgr))


class FakeCourtFinder:
    """Scripted finder for tests: the given point dicts per call, the last repeating."""

    def __init__(self, per_call: list[dict], version: str = "court@fake"):
        self.per_call, self.version, self.calls = per_call, version, 0

    def points(self, frame_bgr) -> dict[str, tuple[float, float]]:
        pts = self.per_call[min(self.calls, len(self.per_call) - 1)]
        self.calls += 1
        return dict(pts)

    def mapping(self, frame_bgr) -> Mapping | None:
        return fit_mapping(self.points(frame_bgr))

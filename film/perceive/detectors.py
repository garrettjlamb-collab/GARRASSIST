from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from film.common import registry

IMGSZ = {"rim": 640, "ball": 320}
RIM_WORLD_CLASSES = {"basketball hoop": "rim", "backboard": "backboard"}
BALL_WORLD_CLASSES = {"basketball": "ball"}
WORLD_WEIGHTS = "yolov8s-worldv2.pt"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: str

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def shifted(self, dx: float, dy: float) -> "Box":
        return replace(self, x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)


class Detector(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray) -> list[Box]: ...


class FakeDetector:
    """Scripted detector for tests: per-call box lists (the last repeats), or a function of (image, call_index)."""

    def __init__(self, per_call: list[list[Box]] | None = None,
                 fn: Callable[[np.ndarray, int], list[Box]] | None = None, name: str = "fake@test"):
        self.per_call = per_call or [[]]
        self.fn = fn
        self.name = name
        self.calls = 0

    def detect(self, image_bgr: np.ndarray) -> list[Box]:
        i = self.calls
        self.calls += 1
        if self.fn is not None:
            return list(self.fn(image_bgr, i))
        return list(self.per_call[min(i, len(self.per_call) - 1)])


def boxes_from_result(result, cls_map: dict[str, str] | None = None) -> list[Box]:
    cls_map = cls_map or {}
    names = result.names
    out = []
    for xyxy, conf, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        raw = str(names[int(k)])
        out.append(Box(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf), cls_map.get(raw, raw)))
    return out


class YoloDetector:
    def __init__(self, model, device: str, imgsz: int, conf: float = 0.25,
                 cls_map: dict[str, str] | None = None, name: str = "yolo"):
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.cls_map = cls_map or {}
        self.name = name

    @classmethod
    def from_weights(cls, weights, device: str, imgsz: int, conf: float = 0.25, name: str = "yolo") -> "YoloDetector":
        from ultralytics import YOLO

        return cls(YOLO(str(weights)), device, imgsz, conf, None, name)

    @classmethod
    def open_vocab(cls, classes: dict[str, str], device: str, imgsz: int, conf: float = 0.15,
                   name: str = "world", weights=None) -> "YoloDetector":
        """YOLO-World with text prompts; `classes` maps prompt -> our class name. Downloads weights on
        first use, into models/base."""
        from ultralytics import YOLOWorld

        model = YOLOWorld(str(weights or registry.base_weights_path(WORLD_WEIGHTS)))
        model.set_classes(list(classes))
        return cls(model, device, imgsz, conf, dict(classes), name)

    def detect(self, image_bgr: np.ndarray) -> list[Box]:
        results = self.model.predict(image_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        return boxes_from_result(results[0], self.cls_map)


def load_detector(kind: str, device: str, profile: str | None = None):
    """The latest trained detector of this kind, else (rims only) a template detector for the camera
    profile, else the open-vocabulary fallback so the pipeline still runs."""
    found = registry.latest(kind)
    if found is not None:
        version, folder = found
        return YoloDetector.from_weights(folder / "best.pt", device, IMGSZ[kind], name=f"{kind}@{version}")
    if kind == "rim" and profile:
        from film.perceive.rim_templates import TemplateRimDetector

        templates = TemplateRimDetector.for_profile(profile)
        if templates is not None:
            return templates
    classes = RIM_WORLD_CLASSES if kind == "rim" else BALL_WORLD_CLASSES
    return YoloDetector.open_vocab(classes, device, IMGSZ[kind], name=f"{kind}@world-fallback")

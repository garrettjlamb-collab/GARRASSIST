from __future__ import annotations

import json
from importlib import resources

import cv2
import numpy as np

from film.perceive.detectors import Box

PROFILE_PACKAGE = "film.perceive.profiles"


def load_rim_profile(profile_name: str) -> dict | None:
    """The `rims` section of a camera profile, or None when the profile has no rim templates."""
    path = resources.files(PROFILE_PACKAGE).joinpath(f"{profile_name.replace('-', '_')}.json")
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("rims")


def _load_gray(name: str) -> np.ndarray:
    data = resources.files(PROFILE_PACKAGE).joinpath(name).read_bytes()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)


def iou(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class TemplateRimDetector:
    """Finds rims for a fixed camera by matching the top of each backboard against a stored template.

    The rim box is a fixed offset below the matched template. Only the top band of the frame is
    searched, and when two templates claim overlapping rims the higher score wins.
    """

    def __init__(self, band: tuple[int, int], templates: list[dict], name: str) -> None:
        self.band = band
        self.templates = templates
        self.name = name

    @classmethod
    def for_profile(cls, profile_name: str) -> "TemplateRimDetector | None":
        rims = load_rim_profile(profile_name)
        if not rims:
            return None
        templates = [
            {
                "name": t["name"],
                "image": _load_gray(t["file"]),
                "threshold": float(t["threshold"]),
                "rim_offset": tuple(t["rim_offset"]),
            }
            for t in rims["templates"]
        ]
        return cls(band=(int(rims["band"][0]), int(rims["band"][1])), templates=templates, name=f"rim@template-{profile_name}")

    def detect(self, image_bgr: np.ndarray) -> list[Box]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        y0, y1 = self.band
        strip = gray[y0:y1]
        found: list[Box] = []
        for t in self.templates:
            tpl = t["image"]
            if strip.shape[0] < tpl.shape[0] or strip.shape[1] < tpl.shape[1]:
                continue
            result = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(result)
            if score < t["threshold"]:
                continue
            ox1, oy1, ox2, oy2 = t["rim_offset"]
            found.append(Box(x + ox1, y + y0 + oy1, x + ox2, y + y0 + oy2, float(score), "rim", side=t["name"]))
        found.sort(key=lambda b: -b.conf)
        kept: list[Box] = []
        for box in found:
            if all(iou(box, other) < 0.3 for other in kept):
                kept.append(box)
        return kept

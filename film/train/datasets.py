from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from film.common.paths import datasets_dir, labels_dir
from film.perceive.detectors import Box

CLASS_NAMES = {"rim": ["rim", "backboard"], "ball": ["ball"]}
VAL_EVERY = 5


# ---- pending queue: what the labeling page shows ----

def pending_dir(dataset: str) -> Path:
    return labels_dir() / dataset / "pending"


def pending_images_dir(dataset: str) -> Path:
    return labels_dir() / dataset / "images"


def write_pending(dataset: str, image_id: str, image_bgr: np.ndarray, boxes: list[Box], extra: dict | None = None) -> Path:
    pending_dir(dataset).mkdir(parents=True, exist_ok=True)
    pending_images_dir(dataset).mkdir(parents=True, exist_ok=True)
    image_path = pending_images_dir(dataset) / f"{image_id}.jpg"
    cv2.imwrite(str(image_path), np.ascontiguousarray(image_bgr))
    h, w = image_bgr.shape[:2]
    data = {"image_id": image_id, "image": str(image_path), "width": w, "height": h,
            "boxes": [{"cls": b.cls, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "conf": b.conf} for b in boxes]}
    if extra:
        data.update(extra)
    path = pending_dir(dataset) / f"{image_id}.json"
    path.write_text(json.dumps(data, indent=1))
    return path


def list_pending(dataset: str) -> list[str]:
    folder = pending_dir(dataset)
    return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []


def read_pending(dataset: str, image_id: str) -> dict:
    return json.loads((pending_dir(dataset) / f"{image_id}.json").read_text())


def remove_pending(dataset: str, image_id: str) -> None:
    for path in (pending_dir(dataset) / f"{image_id}.json", pending_images_dir(dataset) / f"{image_id}.jpg"):
        if path.exists():
            path.unlink()


# ---- YOLO box datasets ----

def yolo_line(cls_idx: int, box: Box, w: int, h: int) -> str:
    cx, cy = (box.x1 + box.x2) / 2 / w, (box.y1 + box.y2) / 2 / h
    return f"{cls_idx} {cx:.6f} {cy:.6f} {box.w / w:.6f} {box.h / h:.6f}"


def data_yaml(dataset: str) -> Path:
    root = datasets_dir() / dataset
    root.mkdir(parents=True, exist_ok=True)
    names = CLASS_NAMES[dataset]
    text = f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    path = root / "data.yaml"
    path.write_text(text)
    return path


def split_for(index: int, val_every: int = VAL_EVERY) -> str:
    return "val" if index % val_every == 0 else "train"


def count_accepted(dataset: str) -> int:
    root = datasets_dir() / dataset / "images"
    return sum(1 for split in ("train", "val") for _ in (root / split).glob("*.jpg")) if root.exists() else 0


def accept_boxes(dataset: str, image_id: str, boxes: list[Box]) -> str:
    """Move a pending image into the YOLO dataset with its final boxes. Returns the split it went to."""
    data = read_pending(dataset, image_id)
    names = CLASS_NAMES[dataset]
    split = split_for(count_accepted(dataset))
    root = datasets_dir() / dataset
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy(data["image"], root / "images" / split / f"{image_id}.jpg")
    lines = [yolo_line(names.index(b.cls), b, data["width"], data["height"]) for b in boxes if b.cls in names]
    (root / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    remove_pending(dataset, image_id)
    return split


# ---- class datasets (ImageFolder layout) ----

def _class_split(dataset: str, cls: str) -> str:
    root = datasets_dir() / dataset
    if not any((root / "val" / cls).glob("*.jpg")):
        return "val"
    total = sum(1 for _ in root.glob("*/*/*.jpg"))
    return split_for(total)


def store_class_image(dataset: str, image_id: str, image_bgr: np.ndarray, cls: str) -> Path:
    split = _class_split(dataset, cls)
    folder = datasets_dir() / dataset / split / cls
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{image_id}.jpg"
    cv2.imwrite(str(path), np.ascontiguousarray(image_bgr))
    return path


def accept_class(dataset: str, image_id: str, cls: str) -> str:
    data = read_pending(dataset, image_id)
    image = cv2.imread(data["image"])
    path = store_class_image(dataset, image_id, image, cls)
    remove_pending(dataset, image_id)
    return path.parts[-3]

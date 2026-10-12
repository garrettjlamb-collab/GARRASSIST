from __future__ import annotations

import json
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np

from film.common.paths import datasets_dir, labels_dir
from film.interpret.court import FLIP_INDEX, POINT_NAMES
from film.perceive.detectors import Box

CLASS_NAMES = {"rim": ["rim", "backboard"], "ball": ["ball"]}
KEYPOINT_NAMES = {"court": POINT_NAMES}        # keypoint datasets: one object class, named points
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

def yolo_line(cls_idx: int, box: Box, w: int, h: int) -> str | None:
    """One YOLO label line, clamped to the image. None for a box under 1 px wide or tall once clamped
    (Ultralytics drops the whole image when a label falls outside [0, 1])."""
    x1, x2 = sorted((min(max(box.x1, 0.0), w), min(max(box.x2, 0.0), w)))
    y1, y2 = sorted((min(max(box.y1, 0.0), h), min(max(box.y2, 0.0), h)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return f"{cls_idx} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def data_yaml(dataset: str) -> Path:
    root = datasets_dir() / dataset
    root.mkdir(parents=True, exist_ok=True)
    if dataset in KEYPOINT_NAMES:
        text = (f"path: {root}\ntrain: images/train\nval: images/val\n"
                f"kpt_shape: [{len(KEYPOINT_NAMES[dataset])}, 3]\nflip_idx: {FLIP_INDEX}\nnames:\n  0: {dataset}\n")
    else:
        names = CLASS_NAMES[dataset]
        text = f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    path = root / "data.yaml"
    path.write_text(text)
    return path


def split_for(index: int, val_every: int = VAL_EVERY) -> str:
    return "val" if index % val_every == 0 else "train"


def split_for_id(image_id: str, val_every: int = VAL_EVERY) -> str:
    """The split is a pure function of the id, so accepting or storing the same image twice never moves it."""
    return "val" if zlib.crc32(image_id.encode()) % val_every == 0 else "train"


def count_accepted(dataset: str) -> int:
    root = datasets_dir() / dataset / "images"
    return sum(1 for split in ("train", "val") for _ in (root / split).glob("*.jpg")) if root.exists() else 0


def _accepted_split(dataset: str, image_id: str) -> str | None:
    root = datasets_dir() / dataset / "images"
    for split in ("train", "val"):
        if (root / split / f"{image_id}.jpg").exists():
            return split
    return None


def skipped_path(dataset: str) -> Path:
    return labels_dir() / dataset / "skipped.txt"


def skipped_ids(dataset: str) -> set[str]:
    path = skipped_path(dataset)
    return set(path.read_text(encoding="utf-8").split()) if path.exists() else set()


def mark_skipped(dataset: str, image_id: str) -> None:
    """Remember that the coach looked at this image and had nothing to label, so it is never queued again."""
    path = skipped_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(image_id + "\n")


def is_known(dataset: str, image_id: str) -> bool:
    """True when this image is waiting for a label, already has one, or was skipped."""
    return ((pending_dir(dataset) / f"{image_id}.json").exists() or _accepted_split(dataset, image_id) is not None
            or image_id in skipped_ids(dataset))


def accept_boxes(dataset: str, image_id: str, boxes: list[Box]) -> str:
    """Move a pending image into the YOLO dataset with its final boxes. Returns the split it went to.
    Idempotent: if the pending entry is gone but the image is already accepted, returns its split."""
    if not (pending_dir(dataset) / f"{image_id}.json").exists():
        done = _accepted_split(dataset, image_id)
        if done is not None:
            return done
    data = read_pending(dataset, image_id)
    names = CLASS_NAMES[dataset]
    split = split_for_id(image_id)
    root = datasets_dir() / dataset
    for other in ("train", "val"):
        if other != split:
            for stale in (root / "images" / other / f"{image_id}.jpg", root / "labels" / other / f"{image_id}.txt"):
                stale.unlink(missing_ok=True)
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy(data["image"], root / "images" / split / f"{image_id}.jpg")
    lines = [ln for b in boxes if b.cls in names
             for ln in [yolo_line(names.index(b.cls), b, data["width"], data["height"])] if ln is not None]
    (root / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    remove_pending(dataset, image_id)
    return split


def accept_points(dataset: str, image_id: str, points: dict[str, tuple[float, float]]) -> str:
    """Move a pending image into a YOLO pose dataset: one object (class 0) whose box bounds the given
    points, and every named keypoint, visible (2) when given, absent (0) otherwise. Returns the split."""
    names = KEYPOINT_NAMES[dataset]
    if not (pending_dir(dataset) / f"{image_id}.json").exists():
        done = _accepted_split(dataset, image_id)
        if done is not None:
            return done
    data = read_pending(dataset, image_id)
    w, h = data["width"], data["height"]
    given = {n: (min(max(x, 0.0), w), min(max(y, 0.0), h)) for n, (x, y) in points.items() if n in names}
    split = split_for_id(image_id)
    root = datasets_dir() / dataset
    for other in ("train", "val"):
        if other != split:
            for stale in (root / "images" / other / f"{image_id}.jpg", root / "labels" / other / f"{image_id}.txt"):
                stale.unlink(missing_ok=True)
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy(data["image"], root / "images" / split / f"{image_id}.jpg")
    if given:
        xs, ys = [p[0] for p in given.values()], [p[1] for p in given.values()]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        fields = [f"{(x1 + x2) / 2 / w:.6f}", f"{(y1 + y2) / 2 / h:.6f}", f"{max(x2 - x1, 1.0) / w:.6f}", f"{max(y2 - y1, 1.0) / h:.6f}"]
        for n in names:
            if n in given:
                fields += [f"{given[n][0] / w:.6f}", f"{given[n][1] / h:.6f}", "2"]
            else:
                fields += ["0.000000", "0.000000", "0"]
        line = "0 " + " ".join(fields) + "\n"
    else:
        line = ""                                                   # a frame with no court in view
    (root / "labels" / split / f"{image_id}.txt").write_text(line)
    remove_pending(dataset, image_id)
    return split


# ---- class datasets (ImageFolder layout) ----

def find_class_images(dataset: str, image_id: str) -> list[Path]:
    root = datasets_dir() / dataset
    return sorted(root.glob(f"*/*/{image_id}.jpg")) if root.exists() else []


def store_class_image(dataset: str, image_id: str, image_bgr: np.ndarray, cls: str) -> Path:
    """Store an image under its class. An id lives in exactly one class folder and one split: an existing copy
    in this class is overwritten in place, copies in other classes are removed. The split is the id's own,
    except that the first image of a class goes to val if val holds none of that class yet, so val always
    has every class."""
    root = datasets_dir() / dataset
    same = [p for p in find_class_images(dataset, image_id) if p.parts[-2] == cls]
    for p in find_class_images(dataset, image_id):
        if p not in same:
            p.unlink()
    if same:
        path = same[0]
    else:
        val_has_cls = any((root / "val" / cls).glob("*.jpg"))
        split = split_for_id(image_id) if val_has_cls else "val"
        path = root / split / cls / f"{image_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(image_bgr))
    return path


def accept_class(dataset: str, image_id: str, cls: str) -> str:
    """Idempotent: with no pending entry, an existing copy is returned (moved to cls if the class changed)."""
    if not (pending_dir(dataset) / f"{image_id}.json").exists():
        existing = find_class_images(dataset, image_id)
        if existing:
            if existing[0].parts[-2] == cls:
                return existing[0].parts[-3]
            return store_class_image(dataset, image_id, cv2.imread(str(existing[0])), cls).parts[-3]
    data = read_pending(dataset, image_id)
    path = store_class_image(dataset, image_id, cv2.imread(data["image"]), cls)
    remove_pending(dataset, image_id)
    return path.parts[-3]

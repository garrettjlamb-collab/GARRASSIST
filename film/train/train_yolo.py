from __future__ import annotations

import shutil
from pathlib import Path

from film.common import registry
from film.common.paths import datasets_dir
from film.perceive.detectors import IMGSZ
from film.train.datasets import data_yaml
from film.train.rimclf import CLASSES, train_rimclf

BASE_WEIGHTS = {"rim": "yolo11n.pt", "ball": "yolo11n.pt"}
MIN_TRAIN, MIN_VAL = 20, 4


def dataset_counts(name: str) -> dict[str, int]:
    root = datasets_dir() / name / "images"
    return {split: (sum(1 for _ in (root / split).glob("*.jpg")) if (root / split).exists() else 0) for split in ("train", "val")}


def _metric(results, key: str):
    box = getattr(results, "box", None)
    value = getattr(box, key, None)
    return float(value) if value is not None else None


def train_detector(name: str, device: str, epochs: int = 50, batch: int = 16, yolo_cls=None, log=print) -> tuple[str, Path]:
    """Fine-tune a small YOLO on datasets/<name> and register the result as models/<name>/vN."""
    counts = dataset_counts(name)
    if counts["train"] < MIN_TRAIN or counts["val"] < MIN_VAL:
        raise ValueError(f"{name}: need at least {MIN_TRAIN} train and {MIN_VAL} val images, have {counts}; label more first")
    yaml_path = data_yaml(name)
    version, out = registry.new_version_dir(name)
    if yolo_cls is None:
        from ultralytics import YOLO as yolo_cls
    log(f"train {name} {version}: {counts} images, {epochs} epochs at {IMGSZ[name]} px on {device}")
    model = yolo_cls(str(registry.base_weights_path(BASE_WEIGHTS[name])))
    results = model.train(data=str(yaml_path), epochs=epochs, imgsz=IMGSZ[name], device=device, batch=batch,
                          project=str(out), name="train", exist_ok=True, verbose=False, plots=False)
    best = out / "train" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training produced no weights at {best}")
    shutil.copy(best, out / "best.pt")
    registry.write_manifest(out, name=name, version=version, base=BASE_WEIGHTS[name], epochs=int(epochs), imgsz=int(IMGSZ[name]),
                            device=str(device), map50=_metric(results, "map50"), train_images=counts["train"], val_images=counts["val"])
    log(f"registered {name} {version}: map50={_metric(results, 'map50')}")
    return version, out


def class_counts(dataset: str) -> dict[str, dict[str, int]]:
    root = datasets_dir() / dataset
    return {split: {cls: sum(1 for _ in (root / split / cls).glob("*.jpg")) for cls in CLASSES} for split in ("train", "val")}


def train_rimclf_version(device: str, epochs: int = 8, log=print, **kwargs) -> tuple[str, Path]:
    counts = class_counts("rimclf")
    empty = [f"{split}/{cls}" for split, per in counts.items() for cls, n in per.items() if n == 0]
    if empty:
        raise ValueError(f"rimclf: these class folders are empty: {', '.join(empty)}; label more first")
    version, out = registry.new_version_dir("rimclf")
    log(f"train rimclf {version}: {counts}, {epochs} epochs on {device}")
    info = train_rimclf(datasets_dir() / "rimclf", out, device, epochs=epochs, **kwargs)
    registry.write_manifest(out, name="rimclf", version=version, device=device, **info)
    log(f"registered rimclf {version}: best val acc {info['best_val_acc']:.3f}")
    return version, out

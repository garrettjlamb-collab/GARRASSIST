import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from film.common import registry
from film.train.train_yolo import class_counts, dataset_counts, train_detector, train_rimclf_version


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))


def _yolo_dataset(root: Path, train=24, val=6):
    for split, n in (("train", train), ("val", val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


class StubYOLO:
    created = []

    def __init__(self, weights):
        StubYOLO.created.append(weights)
        self.kwargs = None

    def train(self, **kwargs):
        self.kwargs = kwargs
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"weights")
        return types.SimpleNamespace(box=types.SimpleNamespace(map50=0.91))


def test_train_detector_registers_a_version(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    _yolo_dataset(tmp_path / "datasets" / "rim")
    assert dataset_counts("rim") == {"train": 24, "val": 6}
    version, folder = train_detector("rim", device="cpu", epochs=3, yolo_cls=StubYOLO, log=lambda *a: None)
    assert version == "v1" and (folder / "best.pt").read_bytes() == b"weights"
    assert StubYOLO.created[-1] == "yolo11n.pt"
    info = registry.read_manifest(folder)
    assert info["map50"] == 0.91 and info["epochs"] == 3 and info["imgsz"] == 640 and info["train_images"] == 24
    assert registry.latest("rim") == ("v1", folder)


def test_train_detector_refuses_tiny_dataset(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    _yolo_dataset(tmp_path / "datasets" / "ball", train=5, val=1)
    with pytest.raises(ValueError):
        train_detector("ball", device="cpu", yolo_cls=StubYOLO)


def test_train_rimclf_version(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    root = tmp_path / "datasets" / "rimclf"
    colours = {"made": (0, 0, 255), "missed": (0, 255, 0), "none": (255, 0, 0)}
    for split, n in (("train", 6), ("val", 2)):
        for cls, colour in colours.items():
            (root / split / cls).mkdir(parents=True)
            for i in range(n):
                img = np.zeros((96, 96, 3), np.uint8)
                img[:] = colour
                cv2.imwrite(str(root / split / cls / f"{i}.jpg"), img)
    assert class_counts("rimclf") == {"train": {"made": 6, "missed": 6, "none": 6}, "val": {"made": 2, "missed": 2, "none": 2}}
    version, folder = train_rimclf_version("cpu", epochs=40, arch="tiny", pretrained=False, lr=0.05, batch=8, log=lambda *a: None)
    assert version == "v1" and (folder / "best.pt").exists()
    assert registry.read_manifest(folder)["arch"] == "tiny"
    assert registry.latest("rimclf") == ("v1", folder)


def test_train_rimclf_refuses_empty_class(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    root = tmp_path / "datasets" / "rimclf"
    for split in ("train", "val"):
        for cls in ("made", "missed", "none"):
            (root / split / cls).mkdir(parents=True)
    cv2.imwrite(str(root / "train" / "made" / "0.jpg"), np.zeros((96, 96, 3), np.uint8))
    with pytest.raises(ValueError, match="missed"):
        train_rimclf_version("cpu", epochs=1, arch="tiny", pretrained=False)

import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from film.common import registry
from film.train.train_yolo import class_counts, dataset_counts, train_detector, train_rimclf_version


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)              # never adopt the repo's own stray weights into a temp dir
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
    assert StubYOLO.created[-1] == str(tmp_path / "models" / "base" / "yolo11n.pt")
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


def test_a_classifier_no_better_than_guessing_is_not_put_into_service(monkeypatch, tmp_path):
    """A model that only matches 'always answer the commonest class' must not be picked up by interpret."""
    _env(monkeypatch, tmp_path)
    import film.train.train_yolo as T
    root = tmp_path / "datasets" / "rimclf"
    for split, counts in (("train", {"made": 2, "missed": 2, "none": 6}), ("val", {"made": 2, "missed": 2, "none": 6})):
        for cls, n in counts.items():
            (root / split / cls).mkdir(parents=True)
            for i in range(n):
                cv2.imwrite(str(root / split / cls / f"{i}.jpg"), np.zeros((32, 32, 3), np.uint8))

    def fake_train(dataset_dir, out_dir, device, epochs=8, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "best.pt").write_bytes(b"weights")
        return {"arch": "tiny", "epochs": epochs, "best_val_acc": fake_train.acc, "train_images": 10, "val_images": 10}

    monkeypatch.setattr(T, "train_rimclf", fake_train)
    lines = []
    fake_train.acc = 0.62                                   # the commonest class is 60% of val: 0.62 is not a real gain
    version, folder = train_rimclf_version("cpu", log=lines.append)
    info = registry.read_manifest(folder)
    assert registry.latest("rimclf") is None and not (folder / "best.pt").exists() and (folder / "not-good-enough.pt").exists()
    assert info["in_service"] is False and info["baseline_val_acc"] == 0.6
    assert any("not put into service" in line for line in lines)
    fake_train.acc = 0.85
    version, folder = train_rimclf_version("cpu", log=lines.append)
    assert registry.latest("rimclf") == (version, folder) and registry.read_manifest(folder)["in_service"] is True

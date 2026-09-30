from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from film.train.rimclf import CLASSES, FakeClassifier, RimClassifier, build_model, train_rimclf

COLOURS = {"made": (0, 0, 255), "missed": (0, 255, 0), "none": (255, 0, 0)}


def make_dataset(root: Path, per_split=(6, 2)):
    for split, n in zip(("train", "val"), per_split):
        for cls, colour in COLOURS.items():
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(n):
                img = np.zeros((96, 96, 3), np.uint8)
                img[:] = colour
                cv2.imwrite(str(folder / f"{i}.jpg"), img)


def test_fake_classifier_cycles_labels():
    f = FakeClassifier([("made", 0.9), ("missed", 0.8)])
    assert f.predict(None)[0] == "made" and f.predict(None)[0] == "missed" and f.predict(None)[0] == "missed"


def test_train_tiny_and_predict(tmp_path):
    torch.manual_seed(0)
    make_dataset(tmp_path / "ds")
    out = tmp_path / "out"
    info = train_rimclf(tmp_path / "ds", out, device="cpu", epochs=40, batch=8, lr=0.05, arch="tiny", pretrained=False)
    assert info["best_val_acc"] == 1.0 and (out / "best.pt").exists()
    assert info["train_images"] == 18 and info["val_images"] == 6

    model = build_model("tiny", pretrained=False)
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    clf = RimClassifier(model, "cpu", version="rimclf@test")
    red = np.zeros((288, 288, 3), np.uint8)
    red[:] = COLOURS["made"]
    label, prob, scores = clf.predict(red)
    assert label == "made" and prob > 0.5 and set(scores) == set(CLASSES)


def test_load_returns_none_without_a_trained_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert RimClassifier.load("cpu") is None


def test_train_refuses_wrong_class_folders(tmp_path):
    (tmp_path / "ds" / "train" / "hit").mkdir(parents=True)
    (tmp_path / "ds" / "val" / "hit").mkdir(parents=True)
    with pytest.raises(ValueError):
        train_rimclf(tmp_path / "ds", tmp_path / "out", device="cpu", epochs=1, arch="tiny", pretrained=False)

import numpy as np
import pytest

from film.common import registry
from film.interpret.rimview import GRID, TILE
from film.train.picclf import PictureClassifier, fit_picclf
from film.train.rimclf import CLASSES

COLOURS = {"made": (0, 0, 220), "missed": (0, 220, 0), "none": (220, 0, 0)}


def sheets(n=6, seed=0):
    rng = np.random.default_rng(seed)
    out, y = [], []
    for k, cls in enumerate(CLASSES):
        for _ in range(n):
            s = np.zeros((GRID * TILE, GRID * TILE, 3), np.uint8)
            s[:] = COLOURS[cls]
            out.append(np.clip(s.astype(int) + rng.integers(-20, 20, s.shape), 0, 255).astype(np.uint8))
            y.append(k)
    return out, np.array(y)


def test_fit_learns_to_tell_the_three_kinds_of_view_apart():
    X, y = sheets()
    clf = fit_picclf(X, y, device="cpu", arch="tiny", pretrained=False, epochs=30, lr=0.02, seed=0)
    probs = clf.scores(X)
    assert probs.shape == (len(X), 3) and np.allclose(probs.sum(axis=1), 1, atol=1e-4)
    assert (probs.argmax(axis=1) == y).mean() == 1.0
    label, prob, scores = clf.predict(X[0])
    assert label == "made" and prob > 0.5 and set(scores) == set(CLASSES)


def test_fit_is_seeded():
    X, y = sheets()
    a = fit_picclf(X, y, device="cpu", arch="tiny", pretrained=False, epochs=3, lr=0.02, seed=1).scores(X[:3])
    b = fit_picclf(X, y, device="cpu", arch="tiny", pretrained=False, epochs=3, lr=0.02, seed=1).scores(X[:3])
    assert np.allclose(a, b, atol=1e-5)


def test_a_saved_classifier_loads_from_the_registry_and_answers_the_same(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert PictureClassifier.load("cpu") is None
    X, y = sheets()
    clf = fit_picclf(X, y, device="cpu", arch="tiny", pretrained=False, epochs=5, lr=0.02, seed=0)
    version, folder = registry.new_version_dir("picclf")
    clf.save(folder / "best.pt")
    loaded = PictureClassifier.load("cpu")
    assert loaded.version == "picclf@v1"
    assert np.allclose(loaded.scores(X[:4]), clf.scores(X[:4]), atol=1e-5)


def test_training_views_can_be_given_as_files(tmp_path):
    import cv2

    X, y = sheets(n=3)
    files = []
    for i, s in enumerate(X):
        files.append(tmp_path / f"{i}.png")
        cv2.imwrite(str(files[-1]), s)
    clf = fit_picclf(files, y, device="cpu", arch="tiny", pretrained=False, epochs=20, lr=0.02, seed=0)
    assert (clf.scores(files).argmax(axis=1) == y).mean() == 1.0


def test_a_view_of_the_wrong_size_is_refused():
    X, y = sheets(n=1)
    clf = fit_picclf(X, y, device="cpu", arch="tiny", pretrained=False, epochs=1, lr=0.02, seed=0)
    with pytest.raises(ValueError, match="view"):
        clf.scores([np.zeros((100, 100, 3), np.uint8)])

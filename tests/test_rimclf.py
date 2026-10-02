import numpy as np
import pytest

from film.common import registry
from film.interpret.ballpath import FEATURES
from film.train.rimclf import CLASSES, FakeClassifier, RimClassifier, fit_rimclf

CENTRES = {"made": [0.5, 1.2, 0.05, 0.0, 0.6, 0.8], "missed": [0.0, 0.1, 0.6, 0.4, 0.6, 0.8], "none": [0.0, -0.5, 1.3, 0.0, 0.05, 0.2]}


def make_rows(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([np.array(CENTRES[c]) + rng.normal(0, 0.03, len(FEATURES)) for c in CLASSES for _ in range(n)])
    y = np.array([CLASSES.index(c) for c in CLASSES for _ in range(n)])
    return X, y


def test_fake_classifier_cycles_labels():
    f = FakeClassifier([("made", 0.9), ("missed", 0.8)])
    assert f.predict(None)[0] == "made" and f.predict(None)[0] == "missed" and f.predict(None)[0] == "missed"


def test_fit_separates_the_three_kinds_of_path():
    X, y = make_rows()
    clf = fit_rimclf(X, y)
    for cls, centre in CENTRES.items():
        label, prob, scores = clf.predict(centre)
        assert label == cls and prob > 0.6 and set(scores) == set(CLASSES) and abs(sum(scores.values()) - 1) < 1e-5


def test_fit_is_deterministic():
    X, y = make_rows()
    assert fit_rimclf(X, y).predict(CENTRES["made"]) == fit_rimclf(X, y).predict(CENTRES["made"])


def test_fit_weighs_classes_so_a_rare_one_is_still_found():
    X, y = make_rows()
    keep = np.r_[np.where(y == 0)[0][:3], np.where(y != 0)[0]]                  # only 3 makes among 40 others
    assert fit_rimclf(X[keep], y[keep]).predict(CENTRES["made"])[0] == "made"


def test_a_saved_classifier_loads_from_the_registry_and_answers_the_same(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert RimClassifier.load() is None
    X, y = make_rows()
    clf = fit_rimclf(X, y)
    version, folder = registry.new_version_dir("rimclf")
    clf.save(folder / "best.pt")
    loaded = RimClassifier.load()
    assert loaded.version == "rimclf@v1"
    assert loaded.predict(CENTRES["missed"])[0] == "missed"
    assert loaded.predict(CENTRES["missed"])[1] == pytest.approx(clf.predict(CENTRES["missed"])[1], abs=1e-5)


def test_weights_from_another_kind_of_classifier_are_not_loaded(tmp_path, monkeypatch):
    """models/rimclf may still hold an image classifier's weights from before the path classifier."""
    import torch
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    _, folder = registry.new_version_dir("rimclf")
    torch.save({"conv.weight": torch.zeros(2, 2)}, folder / "best.pt")
    assert RimClassifier.load() is None

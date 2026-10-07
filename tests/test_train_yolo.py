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


PATHS = {
    "made": {3: (0.0, 0.0, 0.8), 5: (0.02, 0.4, 0.7), 7: (0.0, 0.9, 0.7), 9: (-0.03, 1.3, 0.6)},
    "missed": {3: (0.0, 0.0, 0.8), 6: (0.5, -0.3, 0.7), 9: (0.9, -0.5, 0.7), 12: (1.3, 0.2, 0.6)},
    "none": {},
}


def _rimclf_item(game, cls, i, points=None, with_path=True):
    from film.train import datasets as ds
    from film.train.rimclf_dataset import item_id, save_path

    image_id = item_id(game, f"shot-{cls}-{i:03d}-r1")
    ds.store_class_image("rimclf", image_id, np.zeros((16, 16, 3), np.uint8), cls)
    if with_path:
        points = PATHS[cls] if points is None else points
        frames = [{"dt": round((k - 3) / 15, 3), "balls": [list(points[k])] if k in points else []} for k in range(25)]
        save_path(image_id, {"rim_w": 50.0, "frames": frames}, game, 1, float(i))
    return image_id


def _rimclf_dataset(games=("game-a", "game-b"), per_class=8, points=None):
    for game in games:
        for cls in ("made", "missed", "none"):
            for i in range(per_class):
                _rimclf_item(game, cls, i, points=points)


def test_train_rimclf_registers_a_classifier_that_reads_ball_paths(monkeypatch, tmp_path):
    from film.interpret.ballpath import path_features
    from film.train.rimclf import RimClassifier

    _env(monkeypatch, tmp_path)
    _rimclf_dataset()
    lines = []
    version, folder = train_rimclf_version(log=lines.append)
    info = registry.read_manifest(folder)
    assert version == "v1" and registry.latest("rimclf") == ("v1", folder) and info["in_service"] is True
    assert info["scored_by"] == "leave-one-game-out" and info["accuracy"] == 1.0 and info["items"] == 48
    assert set(info["per_game"]) == {"game-a", "game-b"} and info["made_vs_missed"] == 1.0
    clf = RimClassifier.load()
    made = {"frames": [{"dt": round((k - 3) / 15, 3), "balls": [list(PATHS["made"][k])] if k in PATHS["made"] else []} for k in range(25)]}
    assert clf.predict(path_features(made))[0] == "made"
    assert any("registered rimclf v1" in line for line in lines)


def test_one_game_is_scored_on_its_held_out_split(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    _rimclf_dataset(games=("game-a",), per_class=12)
    version, folder = train_rimclf_version(log=lambda *a: None)
    info = registry.read_manifest(folder)
    assert info["scored_by"] == "val-split" and info["in_service"] is True


def test_train_rimclf_refuses_empty_class(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    for i in range(4):
        _rimclf_item("game-a", "made", i)
        _rimclf_item("game-a", "none", i)
    with pytest.raises(ValueError, match="missed"):
        train_rimclf_version(log=lambda *a: None)


def test_a_montage_without_a_ball_path_is_left_out_and_reported(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    _rimclf_dataset()
    _rimclf_item("game-a", "made", 99, with_path=False)
    lines = []
    version, folder = train_rimclf_version(log=lines.append)
    assert registry.read_manifest(folder)["items"] == 48
    assert any("1 montage" in line and "no ball path" in line for line in lines)


def test_a_classifier_no_better_than_guessing_is_not_put_into_service(monkeypatch, tmp_path):
    """A model that only matches 'always answer the commonest class' must not be picked up by interpret."""
    _env(monkeypatch, tmp_path)
    _rimclf_dataset(points={3: (0.0, 0.0, 0.8)})             # every class shows the same path: nothing to learn
    lines = []
    version, folder = train_rimclf_version(log=lines.append)
    info = registry.read_manifest(folder)
    assert registry.latest("rimclf") is None and not (folder / "best.pt").exists() and (folder / "not-good-enough.pt").exists()
    assert info["in_service"] is False and info["baseline"] == round(1 / 3, 4)
    assert any("not put into service" in line for line in lines)


def test_training_places_the_hoop_per_basket_side_not_per_track(monkeypatch, tmp_path):
    """Each clip here comes from its own short rim track, with the ball arriving half a rim width right of the
    box centre: only pooled by side can the column be found."""
    import json
    from film.train.rimclf_dataset import path_file
    from film.train.train_yolo import rimclf_rows

    _env(monkeypatch, tmp_path)
    shifted = {cls: {k: (dx + 0.5, dy, c) for k, (dx, dy, c) in pts.items()} for cls, pts in PATHS.items()}
    n = 0
    for game in ("game-a", "game-b"):
        for cls in ("made", "missed", "none"):
            for i in range(8):
                image_id = _rimclf_item(game, cls, i, points=shifted[cls])
                stored = json.loads(path_file(image_id).read_text())
                stored.update(rim=100 + n, side="left")
                path_file(image_id).write_text(json.dumps(stored))
                n += 1
    X, y, games, splits = rimclf_rows(log=lambda *a: None)
    from film.interpret.ballpath import FEATURES
    made = X[y == 0][:, FEATURES.index("column_frames")]
    assert (made > 0).all()


VIEW_COLOURS = {"made": (0, 0, 220), "missed": (0, 220, 0), "none": (220, 0, 0)}


def _picclf_dataset(games=("game-a", "game-b"), per_class=5, colours=VIEW_COLOURS):
    from film.interpret.rimview import GRID, TILE
    from film.train.rimclf_dataset import view_file

    for game in games:
        for cls in ("made", "missed", "none"):
            for i in range(per_class):
                image_id = _rimclf_item(game, cls, i)
                view = np.zeros((GRID * TILE, GRID * TILE, 3), np.uint8)
                view[:] = colours[cls]
                view_file(image_id).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(view_file(image_id)), view)


def test_train_picclf_scores_on_a_held_out_game_and_registers_a_model_fit_on_all(monkeypatch, tmp_path):
    from film.train.picclf import PictureClassifier
    from film.train.train_yolo import train_picclf_version

    _env(monkeypatch, tmp_path)
    _picclf_dataset()
    lines = []
    version, folder = train_picclf_version("cpu", hold="game-b", arch="tiny", pretrained=False, epochs=30, lr=0.02, log=lines.append)
    info = registry.read_manifest(folder)
    assert version == "v1" and info["in_service"] is True and info["held_out"] == "game-b"
    assert info["accuracy"] == 1.0 and info["made_vs_missed"] == 1.0 and info["items"] == 30 and info["scored_on"] == 15
    assert registry.latest("picclf") == ("v1", folder) and PictureClassifier.load("cpu") is not None
    assert any("registered picclf v1" in line for line in lines)


def test_train_picclf_holds_out_the_game_labeled_last_unless_told(monkeypatch, tmp_path):
    import os
    from film.train.train_yolo import train_picclf_version

    _env(monkeypatch, tmp_path)
    _picclf_dataset()
    old = 1_600_000_000
    for image in (tmp_path / "datasets" / "rimclf").glob("*/*/game-b--*.jpg"):
        os.utime(image, (old, old))
    version, folder = train_picclf_version("cpu", arch="tiny", pretrained=False, epochs=2, lr=0.02, log=lambda *a: None)
    assert registry.read_manifest(folder)["held_out"] == "game-a"


def test_a_picture_classifier_that_learned_nothing_is_not_put_into_service(monkeypatch, tmp_path):
    from film.train.train_yolo import train_picclf_version

    _env(monkeypatch, tmp_path)
    same = {cls: (90, 90, 90) for cls in VIEW_COLOURS}
    _picclf_dataset(colours=same)
    lines = []
    version, folder = train_picclf_version("cpu", hold="game-b", arch="tiny", pretrained=False, epochs=2, lr=0.02, log=lines.append)
    assert registry.read_manifest(folder)["in_service"] is False and registry.latest("picclf") is None
    assert (folder / "not-good-enough.pt").exists() and any("not put into service" in line for line in lines)

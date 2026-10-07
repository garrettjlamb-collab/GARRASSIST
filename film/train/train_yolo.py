from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from film.common import registry
from film.common.paths import datasets_dir
from film.perceive.detectors import IMGSZ
from film.train.datasets import data_yaml
from film.interpret.ballpath import FEATURES, hoop_offset, path_features
from film.train.rimclf import CLASSES, fit_rimclf

MIN_GAIN_OVER_GUESSING = 0.10     # a classifier must beat the majority-class share of val by this much
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


def rimclf_rows(log=print) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Features, class index, game and split for every stored montage that has its ball path.

    The hoop's offset is measured per game and basket from every path on disk, labeled or still pending:
    it needs no labels, only arrivals."""
    root = datasets_dir() / "rimclf"
    paths = {p.stem: json.loads(p.read_text()) for p in sorted((root / "paths").glob("*.json"))}
    def basket(path: dict) -> tuple:
        """The basket a path belongs to: its side of the court when known, else its rim track."""
        return (path["game"], path.get("side") or path["rim"])

    by_basket = defaultdict(list)
    for path in paths.values():
        by_basket[basket(path)].append(path)
    offsets = {key: hoop_offset(group) for key, group in by_basket.items()}
    rows, missing = [], 0
    for split in ("train", "val"):
        for k, cls in enumerate(CLASSES):
            for image in sorted((root / split / cls).glob("*.jpg")):
                path = paths.get(image.stem)
                if path is None:
                    missing += 1
                    continue
                rows.append((path_features(path, offsets[basket(path)]), k, path["game"], split))
    if missing:
        log(f"rimclf: {missing} montage{'s' if missing != 1 else ''} left out, no ball path stored; "
            f"re-run `film prelabel rimclf <game_id>` on a fresh dataset to include them")
    X = np.array([r[0] for r in rows], dtype=float).reshape(len(rows), len(FEATURES))
    return X, np.array([r[1] for r in rows], dtype=int), [r[2] for r in rows], [r[3] for r in rows]


def train_rimclf_version(device: str | None = None, log=print, **kwargs) -> tuple[str, Path]:
    """Fit the ball-path classifier on every labeled path and register it if it clearly beats guessing.

    With two or more games it is scored the way a new game will meet it: trained on the others, tested on
    the one held out. With one game it is scored on that game's held-out split."""
    X, y, games, splits = rimclf_rows(log)
    games_a, splits_a = np.array(games), np.array(splits)
    empty = [cls for k, cls in enumerate(CLASSES) if not (y == k).any()]
    if empty:
        raise ValueError(f"rimclf: no {', '.join(empty)} clips with a ball path; label more first")
    names = sorted(set(games))
    predicted = np.full(len(y), -1)
    if len(names) >= 2:
        scored_by = "leave-one-game-out"
        for game in names:
            held = games_a == game
            predicted[held] = fit_rimclf(X[~held], y[~held], **kwargs).scores(X[held]).argmax(axis=1)
    else:
        scored_by = "val-split"
        held = splits_a == "val"
        if not held.any() or held.all():
            raise ValueError("rimclf: need clips in both the train and val splits; label more first")
        predicted[held] = fit_rimclf(X[~held], y[~held], **kwargs).scores(X[held]).argmax(axis=1)
    scored = predicted >= 0
    accuracy = float((predicted[scored] == y[scored]).mean())
    # Always answering the commonest class scores this much on the held-out clips. A classifier has to
    # beat that clearly, or interpret is better off with the scoreboard alone.
    baseline = float(np.bincount(y[scored], minlength=len(CLASSES)).max() / scored.sum())
    shots = scored & (y < 2) & (predicted < 2)
    made_vs_missed = float((predicted[shots] == y[shots]).mean()) if shots.any() else None
    per_game = {g: round(float((predicted[scored & (games_a == g)] == y[scored & (games_a == g)]).mean()), 4)
                for g in names if (scored & (games_a == g)).any()}
    in_service = accuracy >= baseline + MIN_GAIN_OVER_GUESSING
    version, out = registry.new_version_dir("rimclf")
    fit_rimclf(X, y, **kwargs).save(out / ("best.pt" if in_service else "not-good-enough.pt"))   # latest() only sees best.pt
    registry.write_manifest(
        out, name="rimclf", version=version, kind="ball-path", features=FEATURES, scored_by=scored_by,
        accuracy=round(accuracy, 4), baseline=round(baseline, 4), in_service=in_service,
        made_vs_missed=None if made_vs_missed is None else round(made_vs_missed, 4), per_game=per_game,
        items=int(len(y)), counts={cls: int((y == k).sum()) for k, cls in enumerate(CLASSES)}, games=names)
    if in_service:
        log(f"registered rimclf {version}: accuracy {accuracy:.3f} by {scored_by} (guessing scores {baseline:.3f})")
    else:
        log(f"rimclf {version} not put into service: accuracy {accuracy:.3f} by {scored_by} does not clearly beat "
            f"guessing ({baseline:.3f}); the scoreboard stays in charge of made-or-missed. Label more games and retrain.")
    return version, out


def picclf_rows(log=print) -> tuple[list[Path], np.ndarray, list[str], list[float]]:
    """View file, class index, game and label time for every stored montage that has its rim view."""
    from film.train.rimclf_dataset import view_file

    root = datasets_dir() / "rimclf"
    views, y, games, stamps, missing = [], [], [], [], 0
    for split in ("train", "val"):
        for k, cls in enumerate(CLASSES):
            for image in sorted((root / split / cls).glob("*.jpg")):
                view = view_file(image.stem)
                if not view.exists():
                    missing += 1
                    continue
                views.append(view)
                y.append(k)
                games.append(image.stem.split("--")[0])
                stamps.append(image.stat().st_mtime)
    if missing:
        log(f"picclf: {missing} montage{'s' if missing != 1 else ''} left out, no rim view stored; "
            f"run `film prelabel rimclf <game_id>` to cut them")
    return views, np.array(y, dtype=int), games, stamps


def train_picclf_version(device: str, hold: str | None = None, log=print, **kwargs) -> tuple[str, Path]:
    """Fit the picture classifier and register it if it clearly beats guessing on a game it did not train on.

    One game is held out for the score (the one labeled last, unless `hold` names another); the model that
    goes into service is then fit on every game. Scoring every game in turn is the experiment's job
    (experiments/rimclf_images.py): it costs about eight minutes a game on the laptop."""
    from film.train.picclf import fit_picclf

    views, y, games, stamps = picclf_rows(log)
    empty = [cls for k, cls in enumerate(CLASSES) if not (y == k).any()]
    if empty:
        raise ValueError(f"picclf: no {', '.join(empty)} clips with a rim view; label more first")
    names = sorted(set(games))
    if len(names) < 2:
        raise ValueError("picclf: needs two reviewed games, one to train on and one to score on")
    if hold is None:
        hold = max(names, key=lambda n: max(s for s, g in zip(stamps, games) if g == n))
    elif hold not in names:
        raise ValueError(f"picclf: no game {hold!r} in the dataset; have {', '.join(names)}")
    held = np.array([g == hold for g in games])
    log(f"train picclf: {len(y)} views from {len(names)} games; scoring on {hold} ({int(held.sum())} views)")
    scorer = fit_picclf([v for v, h in zip(views, held) if not h], y[~held], device, **kwargs)
    prob = scorer.scores([v for v, h in zip(views, held) if h])
    truth = y[held]
    accuracy = float((prob.argmax(axis=1) == truth).mean())
    baseline = float(np.bincount(truth, minlength=len(CLASSES)).max() / len(truth))
    shots = truth < 2
    made_vs_missed = float(((prob[:, 0] > prob[:, 1]) == (truth == 0))[shots].mean()) if shots.any() else None
    in_service = accuracy >= baseline + MIN_GAIN_OVER_GUESSING
    del scorer
    version, out = registry.new_version_dir("picclf")
    fit_picclf(views, y, device, **kwargs).save(out / ("best.pt" if in_service else "not-good-enough.pt"))
    registry.write_manifest(
        out, name="picclf", version=version, kind="rim-view-frames", device=str(device), held_out=hold,
        accuracy=round(accuracy, 4), baseline=round(baseline, 4), in_service=in_service,
        made_vs_missed=None if made_vs_missed is None else round(made_vs_missed, 4),
        items=int(len(y)), scored_on=int(held.sum()), games=names,
        counts={cls: int((y == k).sum()) for k, cls in enumerate(CLASSES)})
    if in_service:
        log(f"registered picclf {version}: accuracy {accuracy:.3f} on {hold}, held out (guessing scores {baseline:.3f}); "
            f"made versus missed {made_vs_missed if made_vs_missed is None else round(made_vs_missed, 3)}")
    else:
        log(f"picclf {version} not put into service: accuracy {accuracy:.3f} on {hold} does not clearly beat guessing "
            f"({baseline:.3f}); the ball-path classifier carries on alone.")
    return version, out

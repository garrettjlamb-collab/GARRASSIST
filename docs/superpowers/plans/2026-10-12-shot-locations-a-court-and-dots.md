# Shot locations, plan A: court finder, hand-placed dots, zones, export

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every shot card shows a half court on which the coach places or nudges the shot's dot; the dot gets a zone and shot type by the tracker's rules; x, y, zone reach the export; and a court-keypoint model is labelled, trained and measured so plan B can place dots automatically.

**Architecture:** Three new pure modules (`film/interpret/court.py` geometry and folding, `film/interpret/zones.py` the tracker's rules, `film/perceive/courtfinder.py` the keypoint model wrapper) plus a `court` dataset in the existing label/accept/train pattern (`film/train/datasets.py`, `film/train/prelabel.py`, `film/train/train_yolo.py`), a `film court <game>` stage writing `games/<id>/court.parquet`, and review/export changes. Plan B (people, release, automatic dots) and plan C (teams, confidence, eval) follow as separate plans.

**Tech Stack:** Python 3.12, `uv`, pytest, numpy, OpenCV (`cv2.findHomography` with RANSAC), pandas/parquet, ultralytics YOLO11 pose (`yolo11n-pose.pt`), FastAPI review app, vanilla JS page.

## Global Constraints

- `src`-style purity as in the repo: `film/interpret/*` and `film/perceive/*` modules take arrays and frames, never paths to games, except the stage runners (`film/perceive/run.py`, `film/interpret/run.py`) and the new `film/perceive/court_run.py`.
- Court coordinates are the tracker's half court: 50 ft wide (x), 47 ft deep (y), baseline at y = 0, basket at (25, 5.25). The far end folds by `y' = 94 - y`, `x' = 50 - x`.
- High-school arc: 19.75 ft from the basket, a plain distance test; a three within 10 ft of the baseline (`y < 10`) is a corner three.
- Every stage is re-runnable with `--force`; every model lives in `models/<name>/vN/` with a manifest, via `film.common.registry`.
- Tests need no weights, internet or GPU. Run `uv run pytest -q` before every commit; all pass.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; push after each commit (`git push`).
- Heavy runs (training, a game's court pass) are started only after telling Garrett how long and getting a go; see the handoff.

---

### Task 1: Court template and folding (`film/interpret/court.py`)

**Files:**
- Create: `film/interpret/court.py`
- Test: `tests/test_court.py`

**Interfaces:**
- Produces: `POINT_NAMES: list[str]` (18 names), `TEMPLATE: dict[str, tuple[float, float]]` (full-court feet, near end first), `fold(x, y) -> tuple[float, float]`, `END_OF_POINT: dict[str, str]` (`"near"` or `"far"`), `FLIP_INDEX: list[int]` (the horizontal-mirror partner of each point, for training augmentation).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_court.py
import math

from film.interpret.court import END_OF_POINT, FLIP_INDEX, POINT_NAMES, TEMPLATE, fold


def test_eighteen_named_points_nine_per_end():
    assert len(POINT_NAMES) == 18 and len(set(POINT_NAMES)) == 18
    assert sum(END_OF_POINT[n] == "near" for n in POINT_NAMES) == 9
    assert all(n in TEMPLATE for n in POINT_NAMES)


def test_template_is_the_tracker_court_with_the_high_school_arc():
    assert TEMPLATE["near-corner-left"] == (0.0, 0.0) and TEMPLATE["near-corner-right"] == (50.0, 0.0)
    assert TEMPLATE["near-lane-left"] == (17.0, 0.0) and TEMPLATE["near-ft-right"] == (33.0, 19.0)
    assert TEMPLATE["near-ft-top"] == (25.0, 25.0)
    x, y = TEMPLATE["near-arc-left"]
    assert y == 0.0 and math.isclose(math.hypot(x - 25.0, y - 5.25), 19.75, abs_tol=1e-6) and x < 25
    fx, fy = TEMPLATE["far-corner-left"]
    assert (fx, fy) == (50.0, 94.0)                         # the far end is the near end rotated about centre court


def test_fold_maps_the_far_end_onto_the_half_court():
    assert fold(10.0, 5.0) == (10.0, 5.0)
    assert fold(40.0, 89.0) == (10.0, 5.0)
    assert fold(25.0, 47.0) == (25.0, 47.0)


def test_flip_index_pairs_left_with_right():
    for i, name in enumerate(POINT_NAMES):
        partner = POINT_NAMES[FLIP_INDEX[i]]
        assert FLIP_INDEX[FLIP_INDEX[i]] == i
        if "left" in name:
            assert partner == name.replace("left", "right")
        elif "right" in name:
            assert partner == name.replace("right", "left")
        else:
            assert partner == name
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_court.py -q`
Expected: `ModuleNotFoundError: No module named 'film.interpret.court'`

- [ ] **Step 3: Implement**

```python
# film/interpret/court.py
"""The court as the tracker draws it, and the eighteen points the court finder looks for.

Coordinates are the tracker's: 50 ft wide (x), 47 ft deep (y) per half, baseline at y = 0, basket at
(25, 5.25). The full court is 94 ft; the far end is the near end rotated about centre court."""
from __future__ import annotations

import math

WIDTH, HALF_DEPTH, FULL_DEPTH = 50.0, 47.0, 94.0
BASKET = (25.0, 5.25)
THREE_RADIUS = 19.75                    # NFHS: the arc meets the baseline, no straight corner segment
LANE_LEFT, LANE_RIGHT, FT_LINE_Y = 17.0, 33.0, 19.0
FT_CIRCLE_RADIUS = 6.0

_ARC_HALF_WIDTH = math.sqrt(THREE_RADIUS ** 2 - BASKET[1] ** 2)       # where the arc meets y = 0

_NEAR = {
    "corner-left": (0.0, 0.0), "corner-right": (WIDTH, 0.0),
    "lane-left": (LANE_LEFT, 0.0), "lane-right": (LANE_RIGHT, 0.0),
    "ft-left": (LANE_LEFT, FT_LINE_Y), "ft-right": (LANE_RIGHT, FT_LINE_Y),
    "ft-top": (BASKET[0], FT_LINE_Y + FT_CIRCLE_RADIUS),
    "arc-left": (BASKET[0] - _ARC_HALF_WIDTH, 0.0), "arc-right": (BASKET[0] + _ARC_HALF_WIDTH, 0.0),
}


def _rotate(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (WIDTH - x, FULL_DEPTH - y)


POINT_NAMES: list[str] = [f"near-{k}" for k in _NEAR] + [f"far-{k}" for k in _NEAR]
TEMPLATE: dict[str, tuple[float, float]] = {**{f"near-{k}": v for k, v in _NEAR.items()},
                                            **{f"far-{k}": _rotate(v) for k, v in _NEAR.items()}}
END_OF_POINT: dict[str, str] = {n: n.split("-", 1)[0] for n in POINT_NAMES}


def _mirror_name(name: str) -> str:
    if "left" in name:
        return name.replace("left", "right")
    if "right" in name:
        return name.replace("right", "left")
    return name


FLIP_INDEX: list[int] = [POINT_NAMES.index(_mirror_name(n)) for n in POINT_NAMES]


def fold(x: float, y: float) -> tuple[float, float]:
    """A full-court point onto the half court of the end it is in: the far end rotates about centre."""
    if y > HALF_DEPTH:
        return (WIDTH - x, FULL_DEPTH - y)
    return (x, y)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_court.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add film/interpret/court.py tests/test_court.py
git commit -m "Court template: eighteen named points on the tracker's court, and folding

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 2: Homography fit and pixel-to-feet mapping (`film/interpret/court.py`)

**Files:**
- Modify: `film/interpret/court.py`
- Test: `tests/test_court.py`

**Interfaces:**
- Produces: `fit_mapping(points: dict[str, tuple[float, float]], reproj_tol_px: float = 8.0) -> Mapping | None` where `Mapping` is a frozen dataclass with `H: np.ndarray (3x3)`, `points_used: int`, `error_px: float`, `conf: float`, `end: str`; `Mapping.to_court(px, py) -> tuple[float, float]` (full-court feet); `Mapping.to_half(px, py) -> tuple[float, float]` (folded); `Mapping.contains(x, y, margin=2.0) -> bool`.
- Consumes: Task 1.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_court.py
import numpy as np
import pytest

from film.interpret.court import Mapping, TEMPLATE, fit_mapping


def _camera(scale=12.0, shear=0.3, ox=300.0, oy=100.0):
    """A made-up camera: feet -> pixels with perspective, for synthetic tests."""
    H = np.array([[scale, shear * scale, ox], [0.0, scale * 0.6, oy], [0.0, 0.004, 1.0]])

    def project(x, y):
        v = H @ np.array([x, y, 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    return project


def test_fit_recovers_the_mapping_from_five_points():
    project = _camera()
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"]
    pts = {n: project(*TEMPLATE[n]) for n in names}
    m = fit_mapping(pts)
    assert isinstance(m, Mapping) and m.points_used == 5 and m.error_px < 0.5 and m.conf > 0.9 and m.end == "near"
    x, y = m.to_court(*project(10.0, 12.0))
    assert abs(x - 10.0) < 0.1 and abs(y - 12.0) < 0.1


def test_fit_needs_four_points_and_drops_an_outlier():
    project = _camera()
    names = ["near-corner-left", "near-corner-right", "near-ft-left"]
    assert fit_mapping({n: project(*TEMPLATE[n]) for n in names}) is None
    names += ["near-ft-right", "near-ft-top", "near-lane-left"]
    pts = {n: project(*TEMPLATE[n]) for n in names}
    pts["near-lane-left"] = (pts["near-lane-left"][0] + 80.0, pts["near-lane-left"][1])      # a bad click
    m = fit_mapping(pts)
    assert m is not None and m.points_used == 5 and m.error_px < 1.0


def test_far_end_points_fold_to_the_half_court():
    project = _camera()
    names = ["far-corner-left", "far-corner-right", "far-ft-left", "far-ft-right", "far-ft-top"]
    m = fit_mapping({n: project(*TEMPLATE[n]) for n in names})
    assert m.end == "far"
    x, y = m.to_half(*project(40.0, 89.0))
    assert abs(x - 10.0) < 0.2 and abs(y - 5.0) < 0.2


def test_contains_tells_court_from_bench():
    project = _camera()
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"]
    m = fit_mapping({n: project(*TEMPLATE[n]) for n in names})
    assert m.contains(*project(25.0, 20.0)) and not m.contains(*project(-6.0, 20.0))
    assert m.contains(*project(-1.0, 20.0))                                  # within the 2 ft margin
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_court.py -q`
Expected: `ImportError: cannot import name 'Mapping'`

- [ ] **Step 3: Implement**

```python
# append to film/interpret/court.py
from dataclasses import dataclass

import cv2
import numpy as np

MIN_POINTS = 4
GOOD_ERROR_PX = 3.0                    # reprojection error at which confidence is full


@dataclass(frozen=True)
class Mapping:
    H: np.ndarray                      # pixels -> full-court feet (homogeneous)
    points_used: int
    error_px: float
    conf: float
    end: str                           # "near" or "far": the end most of the points belong to

    def to_court(self, px: float, py: float) -> tuple[float, float]:
        v = self.H @ np.array([px, py, 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def to_half(self, px: float, py: float) -> tuple[float, float]:
        return fold(*self.to_court(px, py))

    def contains(self, px: float, py: float, margin: float = 2.0) -> bool:
        x, y = self.to_court(px, py)
        return -margin <= x <= WIDTH + margin and -margin <= y <= FULL_DEPTH + margin


def fit_mapping(points: dict[str, tuple[float, float]], reproj_tol_px: float = 8.0) -> Mapping | None:
    """A homography from the named pixel points to the court template, RANSAC against bad points.
    None with fewer than four points, or fewer than four inliers."""
    names = [n for n in points if n in TEMPLATE]
    if len(names) < MIN_POINTS:
        return None
    src = np.array([points[n] for n in names], dtype=np.float64)
    dst = np.array([TEMPLATE[n] for n in names], dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_tol_px)
    if H is None:
        return None
    inliers = [n for n, keep in zip(names, mask.ravel()) if keep]
    if len(inliers) < MIN_POINTS:
        return None
    if len(inliers) < len(names):
        H, _ = cv2.findHomography(np.array([points[n] for n in inliers]), np.array([TEMPLATE[n] for n in inliers]), 0)
    back = np.linalg.inv(H)
    errors = []
    for n in inliers:
        v = back @ np.array([*TEMPLATE[n], 1.0])
        errors.append(float(np.hypot(v[0] / v[2] - points[n][0], v[1] / v[2] - points[n][1])))
    error = float(np.mean(errors))
    conf = min(1.0, len(inliers) / 8.0 + 0.5) * min(1.0, GOOD_ERROR_PX / max(error, GOOD_ERROR_PX))
    end = "far" if sum(END_OF_POINT[n] == "far" for n in inliers) * 2 > len(inliers) else "near"
    return Mapping(H=H, points_used=len(inliers), error_px=error, conf=round(conf, 3), end=end)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_court.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add film/interpret/court.py tests/test_court.py
git commit -m "Court mapping: a RANSAC homography from named points to feet, folded to the half court

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 3: The tracker's zone and shot-type rules (`film/interpret/zones.py`)

**Files:**
- Create: `film/interpret/zones.py`
- Test: `tests/test_zones.py`

**Interfaces:**
- Produces: `is_three(x, y) -> bool`, `zone_of(x, y) -> str` (one of `restricted, paint, mid-left, mid-right, mid-center, corner-3-left, corner-3-right, wing-3-left, wing-3-right, top-3`), `shot_type_of(x, y) -> str` (`rim, three, floater, midrange`), `near_zone_line(x, y, feet=1.5) -> bool`, `ZONE_LABELS: dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_zones.py
from film.interpret.zones import ZONE_LABELS, is_three, near_zone_line, shot_type_of, zone_of


def test_three_is_a_plain_distance_test_on_the_high_school_arc():
    assert not is_three(25.0, 24.9) and is_three(25.0, 25.1)                # 19.75 ft straight out from the basket
    assert is_three(5.0, 3.0) and not is_three(8.0, 3.0)                    # the arc meets the baseline near x = 6


def test_zones_follow_the_tracker():
    assert zone_of(25.0, 6.0) == "restricted"
    assert zone_of(25.0, 15.0) == "paint"
    assert zone_of(10.0, 10.0) == "mid-left" and zone_of(40.0, 10.0) == "mid-right" and zone_of(25.0, 22.0) == "mid-center"
    assert zone_of(3.0, 5.0) == "corner-3-left" and zone_of(47.0, 5.0) == "corner-3-right"
    assert zone_of(8.0, 20.0) == "wing-3-left" and zone_of(42.0, 20.0) == "wing-3-right"
    assert zone_of(25.0, 28.0) == "top-3"
    assert set(ZONE_LABELS) == {"restricted", "paint", "mid-left", "mid-right", "mid-center",
                                "corner-3-left", "corner-3-right", "wing-3-left", "wing-3-right", "top-3"}


def test_shot_type_follows_the_tracker():
    assert shot_type_of(25.0, 6.0) == "rim" and shot_type_of(25.0, 12.0) == "floater"
    assert shot_type_of(25.0, 18.0) == "midrange" and shot_type_of(25.0, 30.0) == "three"


def test_near_zone_line_flags_dots_close_to_the_arc_or_the_paint():
    assert near_zone_line(25.0, 24.0) and near_zone_line(17.5, 10.0)
    assert not near_zone_line(25.0, 15.0) and not near_zone_line(25.0, 35.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_zones.py -q`
Expected: `ModuleNotFoundError: No module named 'film.interpret.zones'`

- [ ] **Step 3: Implement**

```python
# film/interpret/zones.py
"""The shot tracker's zone and shot-type rules (coaching-sim-web/public/shot-tracker/index.html, getZone and
suggestType), with the high-school arc: a plain 19.75 ft distance test, and a three within 10 ft of the
baseline is a corner three. Film shots classify exactly as hand-charted ones will once the tracker moves to
the same arc."""
from __future__ import annotations

import math

from film.interpret.court import BASKET, LANE_LEFT, LANE_RIGHT, THREE_RADIUS

RESTRICTED_RADIUS = 4.0
PAINT_TOP = 19.0
CORNER_Y = 10.0
FLOATER_RADIUS = 9.0

ZONE_LABELS = {
    "restricted": "Restricted Area", "paint": "Paint", "mid-left": "Mid-Range Left",
    "mid-right": "Mid-Range Right", "mid-center": "Mid-Range Center",
    "corner-3-left": "Left Corner 3", "corner-3-right": "Right Corner 3",
    "wing-3-left": "Left Wing 3", "wing-3-right": "Right Wing 3", "top-3": "Top of Key 3",
}


def _polar(x: float, y: float) -> tuple[float, float]:
    dx, dy = x - BASKET[0], y - BASKET[1]
    return math.hypot(dx, dy), math.degrees(math.atan2(dx, dy))


def is_three(x: float, y: float) -> bool:
    return _polar(x, y)[0] > THREE_RADIUS


def zone_of(x: float, y: float) -> str:
    dist, ang = _polar(x, y)
    if dist <= RESTRICTED_RADIUS:
        return "restricted"
    if is_three(x, y):
        if y < CORNER_Y:
            return "corner-3-left" if x < BASKET[0] else "corner-3-right"
        if ang < -35:
            return "wing-3-left"
        if ang > 35:
            return "wing-3-right"
        return "top-3"
    if LANE_LEFT <= x <= LANE_RIGHT and y <= PAINT_TOP:
        return "paint"
    if ang < -25:
        return "mid-left"
    if ang > 25:
        return "mid-right"
    return "mid-center"


def shot_type_of(x: float, y: float) -> str:
    dist, _ = _polar(x, y)
    if dist <= RESTRICTED_RADIUS:
        return "rim"
    if is_three(x, y):
        return "three"
    if dist <= FLOATER_RADIUS and y <= PAINT_TOP:
        return "floater"
    return "midrange"


def near_zone_line(x: float, y: float, feet: float = 1.5) -> bool:
    """Within `feet` of the arc or the paint's edges: the zone could go either way."""
    dist, _ = _polar(x, y)
    if abs(dist - THREE_RADIUS) <= feet:
        return True
    if y <= PAINT_TOP + feet and (abs(x - LANE_LEFT) <= feet or abs(x - LANE_RIGHT) <= feet):
        return True
    return LANE_LEFT - feet <= x <= LANE_RIGHT + feet and abs(y - PAINT_TOP) <= feet
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_zones.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add film/interpret/zones.py tests/test_zones.py
git commit -m "Zones and shot types by the tracker's rules, with the high-school arc

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 4: A keypoint dataset for the court (`film/train/datasets.py`)

**Files:**
- Modify: `film/train/datasets.py` (add after `accept_boxes`)
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces: `KEYPOINT_NAMES = {"court": POINT_NAMES}`, `accept_points(dataset, image_id, points: dict[str, tuple[float, float]]) -> str` (writes a YOLO pose label: one object, class 0, bbox around the visible points, 18 keypoints with visibility 2 for given and 0 for missing), `data_yaml("court")` writing `kpt_shape: [18, 3]` and `flip_idx`.
- Consumes: Task 1 (`POINT_NAMES`, `FLIP_INDEX`).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_datasets.py
def test_accept_points_writes_a_yolo_pose_label(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))
    from film.interpret.court import POINT_NAMES
    img = np.zeros((100, 200, 3), np.uint8)
    ds.write_pending("court", "f1", img, [])
    split = ds.accept_points("court", "f1", {"near-corner-left": (20.0, 80.0), "near-corner-right": (180.0, 80.0),
                                             "near-ft-top": (100.0, 30.0), "near-ft-left": (70.0, 50.0)})
    label = (tmp_path / "datasets" / "court" / "labels" / split / "f1.txt").read_text().split()
    assert label[0] == "0" and len(label) == 5 + 3 * len(POINT_NAMES)
    cx, cy, w, h = map(float, label[1:5])
    assert abs(cx - 0.5) < 1e-6 and abs(w - 0.8) < 1e-6 and abs(cy - 0.55) < 1e-6 and abs(h - 0.5) < 1e-6
    k = POINT_NAMES.index("near-ft-top")
    assert label[5 + 3 * k:8 + 3 * k] == ["0.500000", "0.300000", "2"]
    k = POINT_NAMES.index("far-ft-top")
    assert label[5 + 3 * k:8 + 3 * k] == ["0.000000", "0.000000", "0"]
    assert ds.list_pending("court") == []


def test_court_data_yaml_describes_the_keypoints(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    from film.interpret.court import FLIP_INDEX
    text = ds.data_yaml("court").read_text()
    assert "kpt_shape: [18, 3]" in text and f"flip_idx: {FLIP_INDEX}" in text and "0: court" in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_datasets.py -q`
Expected: `AttributeError: module 'film.train.datasets' has no attribute 'accept_points'`

- [ ] **Step 3: Implement**

```python
# film/train/datasets.py: near the top, after CLASS_NAMES
from film.interpret.court import FLIP_INDEX, POINT_NAMES

KEYPOINT_NAMES = {"court": POINT_NAMES}        # keypoint datasets: one object class, named points


# replace data_yaml with
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


# after accept_boxes
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
```

Also extend `is_known` and `count_accepted` callers: nothing to change, they use `_accepted_split`, which looks at `images/<split>`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_datasets.py -q`
Expected: all pass (the earlier tests plus 2)

- [ ] **Step 5: Commit**

```bash
git add film/train/datasets.py tests/test_datasets.py
git commit -m "A keypoint dataset for the court: YOLO pose labels from named points

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 5: Queue court frames and accept points over the API

**Files:**
- Modify: `film/train/prelabel.py` (add `prelabel_court`), `film/commands/prelabel.py` (add choice `court`), `film/review/app.py` (add `POST /api/label/court/{image_id}/points`, allow `court` in `_dataset`)
- Test: `tests/test_prelabel.py`, `tests/test_review_app.py`

**Interfaces:**
- Produces: `prelabel_court(game_id, n=40, margin_s=30.0, log=print) -> int` (queues `n` evenly spaced frames with `extra={"t": t, "points": {}}`); endpoint body `{"points": {"near-ft-top": [x, y], ...}}` returning `{"split": ...}`; the `court` dataset is accepted by `_dataset`.
- Consumes: Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_prelabel.py
def test_prelabel_court_queues_bare_frames(synthetic_video, tmp_path, monkeypatch):
    from film.train.prelabel import prelabel_court

    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="handheld")
    assert prelabel_court(m.game_id, n=3, margin_s=0.5, log=lambda *a: None) == 3
    ids = ds.list_pending("court")
    assert len(ids) == 3 and all(ds.read_pending("court", i)["points"] == {} for i in ids)
    assert prelabel_court(m.game_id, n=3, margin_s=0.5, log=lambda *a: None) == 0          # never queued twice
```

```python
# append to tests/test_review_app.py
def test_court_points_are_accepted_over_the_api(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    ds.write_pending("court", "c1", np.zeros((100, 200, 3), np.uint8), [], extra={"t": 1.0, "points": {}})
    nxt = client.get("/api/label/court/next").json()
    assert nxt["image_id"] == "c1" and nxt["points"] == {}
    r = client.post("/api/label/court/c1/points", json={"points": {"near-corner-left": [20, 80], "near-corner-right": [180, 80],
                                                                   "near-ft-top": [100, 30], "near-ft-left": [70, 50]}})
    assert r.status_code == 200 and r.json()["split"] in ("train", "val")
    assert client.get("/api/label/court/next").json()["done"] is True
    bad = client.post("/api/label/court/c1/points", json={"points": {"nowhere": [1, 2]}})
    assert bad.status_code == 400
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_prelabel.py tests/test_review_app.py -q`
Expected: `ImportError: cannot import name 'prelabel_court'` and a 404 on `/api/label/court/next`

- [ ] **Step 3: Implement**

```python
# film/train/prelabel.py, after prelabel_rims
def prelabel_court(game_id: str, n: int = 40, margin_s: float = 30.0, log=print) -> int:
    """Evenly spaced frames for the Label court tab, bare: the coach clicks the court's points."""
    m = Manifest.load(game_dir(game_id))
    count = 0
    for t in sample_times(m.duration_s, n, margin_s):
        image_id = f"{game_id}_{int(round(t * 10)):06d}"
        if is_known("court", image_id):
            continue
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        write_pending("court", image_id, frame, [], extra={"t": t, "points": {}})
        count += 1
    log(f"prelabel court: {count} frames queued for {game_id}")
    return count
```

```python
# film/commands/prelabel.py: choices and dispatch
    parser.add_argument("what", choices=["rim", "ball", "rimclf", "court"])
    ...
    parser.add_argument("--n", type=int, default=None, help="how many to queue (rim 150, ball 300, rimclf none-samples 40, court 40)")
    ...
def run(args) -> int:
    if args.what == "court":
        from film.train.prelabel import prelabel_court

        prelabel_court(args.game_id, n=args.n or 40)
        print("next: uv run film review, then open the Label court tab")
        return 0
    # (existing branches follow unchanged)
```

```python
# film/review/app.py
class PointsLabel(BaseModel):
    points: dict[str, tuple[float, float]]


def _dataset(dataset: str) -> str:
    if dataset not in ds.CLASS_NAMES and dataset not in ds.KEYPOINT_NAMES and dataset != "rimclf":
        raise HTTPException(404, f"no dataset {dataset}")
    return dataset


# inside create_app(), after label_boxes
    @app.post("/api/label/{dataset}/{image_id}/points")
    def label_points(dataset: str, image_id: str, body: PointsLabel):
        _dataset(dataset)
        if dataset not in ds.KEYPOINT_NAMES:
            raise HTTPException(400, f"{dataset} is not a keypoint dataset")
        bad = sorted(set(body.points) - set(ds.KEYPOINT_NAMES[dataset]))
        if bad:
            raise HTTPException(400, f"unknown points {bad}")
        _known_image(dataset, image_id)
        try:
            return {"split": ds.accept_points(dataset, image_id, {k: (float(x), float(y)) for k, (x, y) in body.points.items()})}
        except FileNotFoundError:
            raise HTTPException(404, f"no image {image_id} in {dataset}")
```

In `_known_image`, keypoint datasets use `_accepted_split` like box datasets: change the condition to `dataset in ds.CLASS_NAMES or dataset in ds.KEYPOINT_NAMES`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_prelabel.py tests/test_review_app.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add film/train/prelabel.py film/commands/prelabel.py film/review/app.py tests/test_prelabel.py tests/test_review_app.py
git commit -m "film prelabel court, and court points accepted over the review API

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 6: The Label court tab

**Files:**
- Modify: `film/review/static/index.html`
- Test: a browser check (no unit test for the page; the API is tested in Task 5)

**Interfaces:**
- Consumes: `GET /api/label/court/next` (fields `image_id, image_url, width, height, points, remaining`), `POST /api/label/court/{id}/points`, `POST /api/label/court/{id}/skip`.
- Behaviour: the tab lists the 18 point names down the side, highlighting the next one to click; a click on the frame places the highlighted point and moves to the next; **N** skips a point not in view; **Backspace** removes the last placed point; **S** saves; **D** skips the frame; **Z** zooms (reuses the existing zoom class). Placed points draw as labelled crosses.

- [ ] **Step 1: Add the tab and the renderer**

In the `<nav>` add `<button data-tab="court">Label court</button>`. In `render()`, treat `court` like the box tabs (`$("#label").hidden = state.tab === "shots" || state.tab === "quarters"`). In `loadLabel()`: `if (dataset === "rimclf") renderClipLabel(); else if (dataset === "court") renderCourtLabel(); else renderBoxLabel();`. Add:

```javascript
const COURT_POINTS = ["near-corner-left", "near-corner-right", "near-lane-left", "near-lane-right", "near-ft-left", "near-ft-right",
  "near-ft-top", "near-arc-left", "near-arc-right", "far-corner-left", "far-corner-right", "far-lane-left", "far-lane-right",
  "far-ft-left", "far-ft-right", "far-ft-top", "far-arc-left", "far-arc-right"];

function renderCourtLabel() {
  const l = state.label;
  state.points = { ...(l.points || {}) };
  state.pointIndex = COURT_POINTS.findIndex(n => !(n in state.points));
  state.img = null;
  $("#label").innerHTML = `<div class="row"><b>${l.remaining} pending</b><span class="muted">${esc(l.image_id)}</span></div>
    <div class="card wide" style="display:flex;gap:12px;align-items:flex-start">
      <div><canvas id="cv" class="${state.zoom ? "zoom" : ""}" width="${l.width}" height="${l.height}"></canvas>
      <div class="keys">Click where the highlighted point is. <kbd>N</kbd> not in view <kbd>Backspace</kbd> undo
        <kbd>Z</kbd> zoom <kbd>S</kbd> save <kbd>D</kbd> skip frame. "near" is the end nearer the camera.</div></div>
      <div id="pt-list" class="list" style="min-width:200px;max-height:none"></div></div>`;
  const img = new Image();
  img.onload = () => { state.img = img; drawPoints(); };
  img.src = l.image_url;
  $("#cv").onclick = (ev) => {
    if (state.pointIndex < 0) return;
    const p = canvasPos(ev);
    state.points[COURT_POINTS[state.pointIndex]] = [p.x, p.y];
    state.pointIndex = COURT_POINTS.findIndex((n, i) => i > state.pointIndex && !(n in state.points));
    drawPoints();
  };
  drawPoints();
}

function drawPoints() {
  const cv = $("#cv");
  if (!cv) return;
  if (state.img) {
    const ctx = cv.getContext("2d");
    ctx.drawImage(state.img, 0, 0);
    ctx.font = "16px sans-serif"; ctx.lineWidth = 2; ctx.strokeStyle = "#3ddc84"; ctx.fillStyle = "#3ddc84";
    for (const [name, [x, y]] of Object.entries(state.points)) {
      ctx.beginPath(); ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y); ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8); ctx.stroke();
      ctx.fillText(name.replace("near-", "n:").replace("far-", "f:"), x + 10, y - 6);
    }
  }
  $("#pt-list").innerHTML = COURT_POINTS.map((n, i) =>
    `<div class="${i === state.pointIndex ? "sel" : ""}">${n in state.points ? "✓" : state.skippedPoints && state.skippedPoints.has(n) ? "–" : "·"} ${esc(n)}</div>`).join("");
}

async function courtKeys(ev) {
  const l = state.label;
  if (!l || l.done) return;
  const k = ev.key.toLowerCase();
  if (k === "n") {
    state.skippedPoints = state.skippedPoints || new Set();
    if (state.pointIndex >= 0) { state.skippedPoints.add(COURT_POINTS[state.pointIndex]); }
    state.pointIndex = COURT_POINTS.findIndex((n, i) => i > state.pointIndex && !(n in state.points));
    drawPoints();
  } else if (ev.key === "Backspace") {
    const placed = Object.keys(state.points);
    if (placed.length) { const last = placed[placed.length - 1]; delete state.points[last]; state.pointIndex = COURT_POINTS.indexOf(last); drawPoints(); }
  } else if (k === "z") {
    state.zoom = !state.zoom; $("#cv").classList.toggle("zoom", state.zoom);
  } else if (k === "s") {
    await guarded(async () => { await post(`/api/label/court/${l.image_id}/points`, { points: state.points }); state.skippedPoints = null; await loadLabel(); });
  } else if (k === "d") {
    await guarded(async () => { await post(`/api/label/court/${l.image_id}/skip`); state.skippedPoints = null; await loadLabel(); });
  } else return;
  ev.preventDefault();
}
```

In the keydown dispatcher add `else if (state.tab === "court") courtKeys(ev);` before the `boxKeys` fallback, and add `"court"` to the list of tabs a `#hash` may open. The mouseup handler that draws boxes must ignore the court tab: add `state.tab === "court"` to its early-return condition.

- [ ] **Step 2: Check it in a browser**

Run: `uv run film prelabel court <a processed game id> --n 3`, then `uv run film review`, open `http://127.0.0.1:8765/#court`. Click the first few points, press N on one, Backspace once, S to save; the next frame loads and the first one's label file appears under `datasets/court/labels/`. Confirm with:

```bash
ls datasets/court/labels/*/ && head -c 200 datasets/court/labels/*/*.txt
```

Expected: one `.txt` per saved frame, starting with `0 ` and holding 5 + 54 numbers.

- [ ] **Step 3: Commit**

```bash
git add film/review/static/index.html
git commit -m "Label court tab: click the court's named points, N for not in view

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 7: Train the court finder (`film train court`)

**Files:**
- Modify: `film/train/train_yolo.py` (`BASE_WEIGHTS["court"] = "yolo11n-pose.pt"`, `IMGSZ` gains `"court": 960`, `dataset_counts` works unchanged), `film/perceive/detectors.py` (`IMGSZ["court"] = 960`), `film/commands/train.py` (choice `court`)
- Test: `tests/test_train_yolo.py`

**Interfaces:**
- Produces: `train_detector("court", ...)` registering `models/court/vN/best.pt` with a manifest holding `map50` (pose mAP) and `imgsz`. No new function: `train_detector` already generalises; the test proves it with the stub.
- Consumes: Task 4's `data_yaml("court")`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_train_yolo.py
def test_train_court_uses_pose_weights_and_the_keypoint_yaml(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    from film.train import datasets as ds
    from film.interpret.court import POINT_NAMES
    root = tmp_path / "datasets" / "court"
    for split, n in (("train", 24), ("val", 6)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4 " + " ".join(["0.5 0.5 2"] * len(POINT_NAMES)) + "\n")
    StubYOLO.created.clear()
    version, folder = train_detector("court", device="cpu", epochs=1, yolo_cls=StubYOLO, log=lambda *a: None)
    assert version == "v1" and registry.latest("court") == ("v1", folder)
    assert StubYOLO.created[-1].endswith("yolo11n-pose.pt")
    assert "kpt_shape: [18, 3]" in (root / "data.yaml").read_text()
    assert registry.read_manifest(folder)["imgsz"] == 960
```

Read `StubYOLO` at the top of `tests/test_train_yolo.py` first: it records the weights path it was constructed with in `StubYOLO.created` and returns a results object with `box.map50`. If its results lack a `pose` attribute, `_metric(results, "map50")` still reads `box.map50`, which is fine for this test.

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_train_yolo.py -q -k court`
Expected: `KeyError: 'court'`

- [ ] **Step 3: Implement**

```python
# film/perceive/detectors.py
IMGSZ = {"rim": 640, "ball": 640, "court": 960}

# film/train/train_yolo.py
BASE_WEIGHTS = {"rim": "yolo11n.pt", "ball": "yolo11n.pt", "court": "yolo11n-pose.pt"}

# film/commands/train.py: choices
    parser.add_argument("what", choices=["rim", "ball", "rimclf", "picclf", "court"])
    # the else-branch already calls train_detector(args.what, ...)
```

In `train_detector`, after `results = model.train(...)`, prefer the pose metric when present:

```python
    score = _metric(getattr(results, "pose", None) and results, "map50")   # keep as is: _metric reads results.box
```

Leave `_metric` unchanged; ultralytics pose results carry `box.map50` as well, and the handoff notes that the pose keypoint metric is read from `results.pose.map50` by hand when needed.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_train_yolo.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add film/perceive/detectors.py film/train/train_yolo.py film/commands/train.py tests/test_train_yolo.py
git commit -m "film train court: a YOLO pose model on the court's eighteen points

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 8: The court finder wrapper (`film/perceive/courtfinder.py`)

**Files:**
- Create: `film/perceive/courtfinder.py`
- Test: `tests/test_courtfinder.py`

**Interfaces:**
- Produces: `class CourtFinder` with `points(frame_bgr) -> dict[str, tuple[float, float]]` (named points with keypoint confidence >= `MIN_KPT_CONF = 0.5`), `mapping(frame_bgr) -> Mapping | None`, `version: str`; `CourtFinder.load(device) -> CourtFinder | None` from `registry.latest("court")`; `FakeCourtFinder(points_per_call)` for tests.
- Consumes: Task 2 (`fit_mapping`), Task 1 (`POINT_NAMES`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_courtfinder.py
import types

import numpy as np

from film.interpret.court import POINT_NAMES, TEMPLATE
from film.perceive.courtfinder import CourtFinder, FakeCourtFinder, points_from_result


def _result(xy, conf):
    kp = types.SimpleNamespace(xy=np.array([xy]), conf=np.array([conf]))
    boxes = types.SimpleNamespace(conf=np.array([0.9]))
    return types.SimpleNamespace(keypoints=kp, boxes=boxes)


def test_points_from_result_keeps_confident_named_points():
    xy = [[10.0 * i, 5.0 * i] for i in range(len(POINT_NAMES))]
    conf = [0.9] * len(POINT_NAMES)
    conf[3] = 0.1
    pts = points_from_result(_result(xy, conf))
    assert len(pts) == len(POINT_NAMES) - 1 and POINT_NAMES[3] not in pts and pts[POINT_NAMES[0]] == (0.0, 0.0)
    assert points_from_result(types.SimpleNamespace(keypoints=None, boxes=None)) == {}


def test_court_finder_maps_a_frame_when_enough_points_are_confident():
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"]
    pix = {n: (TEMPLATE[n][0] * 10 + 100, 500 - TEMPLATE[n][1] * 10) for n in names}
    finder = FakeCourtFinder([pix, {}])
    m = finder.mapping(np.zeros((8, 8, 3), np.uint8))
    assert m is not None and m.points_used == 5
    x, y = m.to_half(100 + 250, 500 - 100)
    assert abs(x - 25.0) < 0.1 and abs(y - 10.0) < 0.1
    assert finder.mapping(np.zeros((8, 8, 3), np.uint8)) is None


def test_load_returns_none_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert CourtFinder.load("cpu") is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_courtfinder.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# film/perceive/courtfinder.py
"""Finds the court's named points in a frame and fits the pixels-to-feet mapping (film.interpret.court)."""
from __future__ import annotations

import numpy as np

from film.common import registry
from film.interpret.court import POINT_NAMES, Mapping, fit_mapping
from film.perceive.detectors import IMGSZ

MIN_KPT_CONF = 0.5


def points_from_result(result) -> dict[str, tuple[float, float]]:
    """Named points from an ultralytics pose result, the most confident court instance only."""
    kp = getattr(result, "keypoints", None)
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return {}
    best = int(np.argmax(np.asarray(result.boxes.conf))) if result.boxes is not None and len(result.boxes.conf) else 0
    xy = np.asarray(kp.xy[best])
    conf = np.asarray(kp.conf[best]) if kp.conf is not None else np.ones(len(xy))
    return {name: (float(x), float(y)) for name, (x, y), c in zip(POINT_NAMES, xy, conf) if c >= MIN_KPT_CONF}


class CourtFinder:
    def __init__(self, model, device: str, imgsz: int, version: str = "court@test"):
        self.model, self.device, self.imgsz, self.version = model, device, imgsz, version

    @classmethod
    def load(cls, device: str) -> "CourtFinder | None":
        found = registry.latest("court")
        if found is None:
            return None
        version, folder = found
        from ultralytics import YOLO

        info = registry.read_manifest(folder) if (folder / "manifest.json").exists() else {}
        return cls(YOLO(str(folder / "best.pt")), device, int(info.get("imgsz") or IMGSZ["court"]), version=f"court@{version}")

    def points(self, frame_bgr: np.ndarray) -> dict[str, tuple[float, float]]:
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, device=self.device, verbose=False, conf=0.25)
        return points_from_result(results[0]) if results else {}

    def mapping(self, frame_bgr: np.ndarray) -> Mapping | None:
        return fit_mapping(self.points(frame_bgr))


class FakeCourtFinder:
    """Scripted finder for tests: the given point dicts per call, the last repeating."""

    def __init__(self, per_call: list[dict], version: str = "court@fake"):
        self.per_call, self.version, self.calls = per_call, version, 0

    def points(self, frame_bgr) -> dict[str, tuple[float, float]]:
        pts = self.per_call[min(self.calls, len(self.per_call) - 1)]
        self.calls += 1
        return dict(pts)

    def mapping(self, frame_bgr) -> Mapping | None:
        return fit_mapping(self.points(frame_bgr))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_courtfinder.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add film/perceive/courtfinder.py tests/test_courtfinder.py
git commit -m "Court finder: named points from the pose model, fitted to a mapping

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 9: The `film court` stage (`film/perceive/court_run.py`, `games/<id>/court.parquet`)

**Files:**
- Create: `film/perceive/court_run.py`, `film/commands/court.py`
- Modify: `film/cli.py` (register `film.commands.court`)
- Test: `tests/test_court_run.py`

**Interfaces:**
- Produces: `court_pass(game_id, finder=None, device=None, step_s=None, log=print) -> dict` writing `court.parquet` with columns `frame, t, end, points_used, error_px, conf, h00..h22` (the nine homography entries) and marking stage `court` in the manifest with `fitted`, `frames`, `step_s`; `load_mappings(gd) -> list[tuple[float, Mapping]]` sorted by t; `mapping_at(mappings, t, max_gap_s=2.0) -> Mapping | None` (nearest fitted frame within the gap).
- Step: `step_s` defaults to 1.0 on `hudl-*` profiles (digital pan over a fixed panorama) and 0.5 otherwise.
- Consumes: Task 8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_court_run.py
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.court import TEMPLATE
from film.perceive.court_run import court_pass, load_mappings, mapping_at
from film.perceive.courtfinder import FakeCourtFinder


def _pix(names):
    return {n: (TEMPLATE[n][0] * 10 + 100, 500 - TEMPLATE[n][1] * 10) for n in names}


def test_court_pass_writes_mappings_once_a_second_and_the_manifest(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="hudl-mdn", fps=10.0)           # 3 s of film
    gd = game_dir(m.game_id)
    good = _pix(["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"])
    finder = FakeCourtFinder([good, {}, good])
    info = court_pass(m.game_id, finder=finder, log=lambda *a: None)
    assert info["status"] == "done" and info["step_s"] == 1.0 and info["frames"] == 3 and info["fitted"] == 2
    maps = load_mappings(gd)
    assert [round(t) for t, _ in maps] == [0, 2]
    assert mapping_at(maps, 0.4) is not None and mapping_at(maps, 1.0) is not None       # the gap is bridged
    assert mapping_at(maps, 1.0).end == "near"
    assert mapping_at(maps, 5.0) is None                                                 # three seconds past the last fit
    assert Manifest.load(gd).model_versions["court"] == "court@fake"


def test_handheld_film_is_fitted_twice_a_second(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="handheld", fps=10.0)
    finder = FakeCourtFinder([_pix(["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right"])])
    info = court_pass(m.game_id, finder=finder, log=lambda *a: None)
    assert info["step_s"] == 0.5 and info["frames"] == 6 and info["fitted"] == 6
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_court_run.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# film/perceive/court_run.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.interpret.court import Mapping
from film.perceive.courtfinder import CourtFinder

COLUMNS = ["frame", "t", "end", "points_used", "error_px", "conf"] + [f"h{i}{j}" for i in range(3) for j in range(3)]
FILE = "court.parquet"
MAX_GAP_S = 2.0


def default_step(profile: str) -> float:
    return 1.0 if profile.startswith("hudl") else 0.5


def court_pass(game_id: str, finder=None, device: str | None = None, step_s: float | None = None, log=print) -> dict:
    """Fit the court mapping every step_s seconds and write court.parquet."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    finder = finder or CourtFinder.load(pick_device(device))
    if finder is None:
        raise RuntimeError("no court model trained yet: label frames in the Label court tab, then `film train court`")
    step = step_s or default_step(m.profile)
    rows, n, started = [], 0, time.time()
    for t, frame in iter_frames(Path(m.source), fps=1.0 / step):
        n += 1
        mapping = finder.mapping(frame)
        if mapping is None:
            continue
        rows.append({"frame": int(round(t * m.fps)), "t": float(t), "end": mapping.end, "points_used": mapping.points_used,
                     "error_px": mapping.error_px, "conf": mapping.conf, **{f"h{i}{j}": float(mapping.H[i, j]) for i in range(3) for j in range(3)}})
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(gd / FILE, index=False)
    m.model_versions["court"] = finder.version
    m.mark("court", "done", frames=n, fitted=len(rows), step_s=step, seconds=round(time.time() - started, 1))
    m.save(gd)
    log(f"court {game_id}: {len(rows)} of {n} frames fitted ({step} s apart) with {finder.version}")
    return m.stages["court"]


def load_mappings(gd: Path) -> list[tuple[float, Mapping]]:
    path = Path(gd) / FILE
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    out = []
    for r in df.itertuples():
        H = np.array([[getattr(r, f"h{i}{j}") for j in range(3)] for i in range(3)])
        out.append((float(r.t), Mapping(H=H, points_used=int(r.points_used), error_px=float(r.error_px), conf=float(r.conf), end=str(r.end))))
    return sorted(out, key=lambda p: p[0])


def mapping_at(mappings: list[tuple[float, Mapping]], t: float, max_gap_s: float = MAX_GAP_S) -> Mapping | None:
    """The fitted mapping nearest in time, if within max_gap_s."""
    if not mappings:
        return None
    best = min(mappings, key=lambda p: abs(p[0] - t))
    return best[1] if abs(best[0] - t) <= max_gap_s else None
```

```python
# film/commands/court.py
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("court", help="fit the court mapping through a game and write court.parquet")
    parser.add_argument("game_id")
    parser.add_argument("--step-s", type=float, default=None, help="seconds between fits (hudl 1.0, else 0.5)")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.perceive.court_run import court_pass

    court_pass(args.game_id, device=args.device, step_s=args.step_s)
    return 0
```

Register `"film.commands.court"` in `film/cli.py`'s module list after `"film.commands.perceive"`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_court_run.py tests/test_cli.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add film/perceive/court_run.py film/commands/court.py film/cli.py tests/test_court_run.py
git commit -m "film court: the court mapping through a game, once a second, in court.parquet

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 10: Location on events and decisions, the zone, and the queue

**Files:**
- Modify: `film/interpret/events.py` (Event gains `x, y, zone, shot_type, location_conf`, all optional, round-tripped by `to_json`/`from_json`), `film/review/decisions.py` (`save_decision` accepts `x, y`; `apply_decisions` overrides `x, y` and recomputes `zone` and `shot_type` from the coach's dot), `film/review/app.py` (`Decision` gains `x, y`)
- Test: `tests/test_events.py`, `tests/test_decisions.py`, `tests/test_review_app.py`

**Interfaces:**
- Produces: on an event dict, `x`, `y` (feet or null), `zone`, `shot_type`, `location_conf`; on a decision, `x`, `y`; `apply_decisions` sets `zone`/`shot_type` from `zones.zone_of`/`shot_type_of` whenever `x, y` are present.
- Consumes: Task 3.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_events.py
def test_an_event_carries_a_location_and_round_trips(tmp_path):
    e = Event(id="shot-000100-r1", type="shot", t=10.0, rim_id=1, quarter=1, clock_s=100.0, team=None, made=True,
              confidence=0.5, clip={"start": 2.0, "end": 13.0}, signals={}, flags=[], x=12.5, y=8.0, zone="mid-left",
              shot_type="midrange", location_conf=0.8)
    save_events(tmp_path, [e])
    [back] = load_events(tmp_path)
    assert (back.x, back.y, back.zone, back.shot_type, back.location_conf) == (12.5, 8.0, "mid-left", "midrange", 0.8)
    old = Event.from_json({k: v for k, v in e.to_json().items() if k not in ("x", "y", "zone", "shot_type", "location_conf")})
    assert old.x is None and old.zone is None
```

```python
# append to tests/test_decisions.py
def test_the_coachs_dot_overrides_the_location_and_sets_the_zone(tmp_path):
    save_decision(tmp_path, "shot-1", {"status": "confirmed", "made": True, "x": 3.0, "y": 4.0})
    save_decision(tmp_path, "shot-2", {"status": "confirmed", "made": True})
    events = [{"id": "shot-1", "made": True, "team": None, "x": 20.0, "y": 20.0, "zone": "mid-center", "shot_type": "midrange"},
              {"id": "shot-2", "made": True, "team": None, "x": 25.0, "y": 6.0, "zone": "restricted", "shot_type": "rim"}]
    merged = apply_decisions(events, load_decisions(tmp_path))
    assert (merged[0]["x"], merged[0]["y"], merged[0]["zone"], merged[0]["shot_type"]) == (3.0, 4.0, "corner-3-left", "three")
    assert (merged[1]["x"], merged[1]["zone"]) == (25.0, "restricted")


def test_a_dot_off_the_court_is_refused(tmp_path):
    with pytest.raises(ValueError, match="court"):
        save_decision(tmp_path, "shot-1", {"status": "confirmed", "x": 60.0, "y": 4.0})
```

```python
# append to tests/test_review_app.py
def test_a_dot_is_saved_with_the_decision(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    r = client.post(f"/api/games/{game_id}/decisions/shot-000000-r1", json={"status": "confirmed", "made": True, "x": 10.0, "y": 12.0})
    assert r.status_code == 200 and r.json()["x"] == 10.0
    e = {x["id"]: x for x in client.get(f"/api/games/{game_id}/events").json()}["shot-000000-r1"]
    assert (e["x"], e["y"], e["zone"]) == (10.0, 12.0, "mid-left")
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_events.py tests/test_decisions.py tests/test_review_app.py -q`
Expected: `TypeError: Event.__init__() got an unexpected keyword argument 'x'` and the others fail

- [ ] **Step 3: Implement**

In `film/interpret/events.py`, add to the `Event` dataclass (after `needs_location`): `x: float | None = None`, `y: float | None = None`, `zone: str | None = None`, `shot_type: str | None = None`, `location_conf: float | None = None`. Check `to_json`/`from_json`: if they use `asdict` and `cls(**{k: v for k in known})`, nothing else changes; if `from_json` lists fields, add these five with defaults.

```python
# film/review/decisions.py
from film.interpret.court import HALF_DEPTH, WIDTH
from film.interpret.zones import shot_type_of, zone_of

# in save_decision, after the foul check
    if ("x" in decision) != ("y" in decision):
        raise ValueError("a dot needs both x and y")
    if "x" in decision and decision["x"] is not None:
        x, y = float(decision["x"]), float(decision["y"])
        if not (-2.0 <= x <= WIDTH + 2.0 and -2.0 <= y <= HALF_DEPTH + 2.0):
            raise ValueError(f"the dot ({x:.1f}, {y:.1f}) is off the court")
        entry["x"], entry["y"] = x, y
    elif "x" in decision:
        entry["x"] = entry["y"] = None

# in apply_decisions, inside `if decision:`
            if "x" in decision:
                merged["x"], merged["y"] = decision["x"], decision["y"]
        if merged.get("x") is not None and merged.get("y") is not None:
            merged["zone"], merged["shot_type"] = zone_of(merged["x"], merged["y"]), shot_type_of(merged["x"], merged["y"])
```

Note `apply_decisions` must set the zone from the event's own `x, y` too (the second assertion), so the `if merged.get("x")` block sits outside `if decision:`.

In `film/review/app.py`, `Decision` gains `x: float | None = None` and `y: float | None = None`; the endpoint already passes `decision.model_dump(exclude_unset=True)`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add film/interpret/events.py film/review/decisions.py film/review/app.py tests/test_events.py tests/test_decisions.py tests/test_review_app.py
git commit -m "A shot carries a location; the coach's dot overrides it and sets the zone

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 11: The half court on the shot card

**Files:**
- Modify: `film/review/static/index.html`
- Test: browser check

**Interfaces:**
- Consumes: event fields `x, y, zone, shot_type, location_conf, flags` and the decision endpoint's `x, y`.
- Behaviour: a 250x235 px canvas of the half court (baseline at the bottom, basket near the bottom centre, the 19.75 ft arc, the lane, the free-throw circle) beside the clip. The dot is drawn when `x, y` exist. **Click** places it; **arrow keys** nudge it one foot; **Shift+arrows** five feet. The zone label and shot type show under it. A card with no dot shows "No dot yet: click the court". Enter/M/X/W save as now; the dot goes in the decision when the coach set or moved it (`e.x`/`e.y` changed from what the server sent). Queue order: among undecided strong cards, those with no dot first.

- [ ] **Step 1: Implement**

Add a helper and drawing code:

```javascript
const HALF = { w: 50, h: 47, basketX: 25, basketY: 5.25, arc: 19.75, laneL: 17, laneR: 33, ftY: 19, ftR: 6, restricted: 4 };
const CV = { w: 250, h: 235 };
const toPx = (x, y) => [x / HALF.w * CV.w, CV.h - y / HALF.h * CV.h];
const toFt = (px, py) => [px / CV.w * HALF.w, (CV.h - py) / CV.h * HALF.h];

function drawHalfCourt(cv, e) {
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#1b2130"; ctx.fillRect(0, 0, CV.w, CV.h);
  ctx.strokeStyle = "#8b95a7"; ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, CV.w - 1, CV.h - 1);
  const [lx, ly] = toPx(HALF.laneL, HALF.ftY), [rx] = toPx(HALF.laneR, 0);
  ctx.strokeRect(lx, ly, rx - lx, CV.h - ly);
  const [bx, by] = toPx(HALF.basketX, HALF.basketY);
  const r = (ft) => ft / HALF.w * CV.w;
  ctx.beginPath(); ctx.arc(bx, by, r(HALF.arc), Math.PI, 2 * Math.PI); ctx.stroke();               // the arc, meeting the baseline
  const [fx, fy] = toPx(HALF.basketX, HALF.ftY);
  ctx.beginPath(); ctx.arc(fx, fy, r(HALF.ftR), 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.arc(bx, by, r(0.75), 0, 2 * Math.PI); ctx.strokeStyle = "#ff8c42"; ctx.stroke();
  if (e.x != null && e.y != null) {
    const [dx, dy] = toPx(e.x, e.y);
    ctx.fillStyle = e.made ? "#3ddc84" : "#ff5c5c";
    ctx.beginPath(); ctx.arc(dx, dy, 6, 0, 2 * Math.PI); ctx.fill();
  }
}
```

In `renderShots()`, inside the card after the video, add:

```javascript
      <div style="display:inline-block;vertical-align:top;margin-left:12px">
        <canvas id="court" width="${CV.w}" height="${CV.h}" style="border-radius:6px;cursor:crosshair"></canvas>
        <div class="muted" id="zone-line">${e.x != null ? esc(ZONE_LABELS[e.zone] || e.zone || "") + " · " + esc(e.shot_type || "") + (e.location_conf != null ? " · dot conf " + e.location_conf : "") : "No dot yet: click the court"}</div>
      </div>
```

with `const ZONE_LABELS = {restricted:"Restricted Area", paint:"Paint", "mid-left":"Mid-Range Left", "mid-right":"Mid-Range Right", "mid-center":"Mid-Range Center", "corner-3-left":"Left Corner 3", "corner-3-right":"Right Corner 3", "wing-3-left":"Left Wing 3", "wing-3-right":"Right Wing 3", "top-3":"Top of Key 3"};` near the top. After the innerHTML assignment: `drawHalfCourt($("#court"), e); $("#court").onclick = (ev) => { const r = $("#court").getBoundingClientRect(); const [x, y] = toFt(ev.clientX - r.left, ev.clientY - r.top); setDot(e, x, y); };` and

```javascript
function setDot(e, x, y) {
  e.x = Math.max(0, Math.min(HALF.w, Math.round(x * 10) / 10)); e.y = Math.max(0, Math.min(HALF.h, Math.round(y * 10) / 10));
  e.dotMoved = true;
  renderShots();
}
```

In `shotKeys`, before the `else return`: arrows nudge when a dot exists: `else if (ev.key.startsWith("Arrow") && e.x != null) { const s = ev.shiftKey ? 5 : 1; setDot(e, e.x + (ev.key === "ArrowRight" ? s : ev.key === "ArrowLeft" ? -s : 0), e.y + (ev.key === "ArrowUp" ? s : ev.key === "ArrowDown" ? -s : 0)); }`. In `decide()` and `confirmHigh()`, add `...(e.dotMoved ? { x: e.x, y: e.y } : {})` to the posted body. The zone shown under the court is recomputed on the next render from the server after saving; before saving, compute it client-side with a small port of `zone_of` for the label (or show "zone updates on save": pick the port, 15 lines, mirroring `film/interpret/zones.py`).

Queue order: in `sortQueue()`, add `((a.x == null) - (b.x == null)) * -1` as a tie-break after the weak/strong ordering so cards without a dot come first among strong undecided cards.

- [ ] **Step 2: Check it in a browser**

Open the review page, pick a reviewed game, click on the half court, press arrow keys, press Enter; reload; the dot is where you left it and the zone label matches. Click the court on a decided card, Enter: the decision updates.

- [ ] **Step 3: Commit**

```bash
git add film/review/static/index.html
git commit -m "Shot card: a half court with the dot; click places it, arrows nudge it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 12: Location in the export and in `film eval`

**Files:**
- Modify: `film/export/shots.py` (FIELDS gain `x, y, zone, shot_type, location_conf`), `film/export/evaluate.py` (new numbers), `film/commands/eval.py` if it prints keys by name
- Test: `tests/test_export_shots.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces: CSV/JSON rows with `x, y, zone, shot_type, location_conf`; `evaluate()` result gains `located` (confirmed shots with a dot), `dot_placed_by_coach` (decisions carrying x), `median_dot_error_ft` (over confirmed shots where both the event and the decision have a dot: distance between the machine's dot and the coach's), `zone_agreement` (share of those where zones match). With no machine dots yet (plan A), the error fields are null.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_export_shots.py
def test_the_export_carries_the_dot_and_zone(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed", "made": False, "x": 3.0, "y": 4.0})
    csv_path, json_path = export_shots(game_id)
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["x"] == "3.0" and rows[0]["zone"] == "corner-3-left" and rows[0]["shot_type"] == "three"
    assert rows[1]["x"] == "" and rows[1]["zone"] == ""
    shots = json.loads(json_path.read_text())["shots"]
    assert shots[0]["x"] == 3.0 and shots[1]["x"] is None
```

```python
# append to tests/test_evaluate.py (read its _game helper first and reuse it)
def test_eval_reports_dot_error_against_the_coach(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    from film.interpret.events import load_events, save_events
    events = load_events(gd)
    events[0].x, events[0].y, events[0].zone = 10.0, 10.0, "mid-left"
    save_events(gd, events)
    save_decision(gd, events[0].id, {"status": "confirmed", "made": True, "x": 13.0, "y": 14.0})
    result = evaluate(game_id)
    assert result["located"] >= 1 and result["dot_placed_by_coach"] == 1
    assert abs(result["median_dot_error_ft"] - 5.0) < 1e-6 and result["zone_agreement"] == 1.0
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_export_shots.py tests/test_evaluate.py -q`
Expected: KeyError on `x` / `located`

- [ ] **Step 3: Implement**

```python
# film/export/shots.py
FIELDS = ["id", "t", "quarter", "clock_s", "team", "made", "confidence", "status", "flags", "foul", "counts",
          "x", "y", "zone", "shot_type", "location_conf"]
# in the writerow dict add:
                             "x": "" if e.get("x") is None else e["x"], "y": "" if e.get("y") is None else e["y"],
                             "zone": e.get("zone") or "", "shot_type": e.get("shot_type") or "",
                             "location_conf": "" if e.get("location_conf") is None else e["location_conf"]
```

```python
# film/export/evaluate.py, inside evaluate() after `compared` is built
    from statistics import median
    from film.interpret.zones import zone_of
    located = [e for e in by_status["confirmed"] if (decisions[e.id].get("x") if "x" in decisions[e.id] else e.x) is not None]
    placed = [e for e in by_status["confirmed"] if decisions[e.id].get("x") is not None]
    pairs = [(e, decisions[e.id]) for e in placed if e.x is not None and e.y is not None]
    errors = [math.hypot(e.x - d["x"], e.y - d["y"]) for e, d in pairs]
    zones_ok = [zone_of(e.x, e.y) == zone_of(d["x"], d["y"]) for e, d in pairs]
    result.update({"located": len(located), "dot_placed_by_coach": len(placed),
                   "median_dot_error_ft": median(errors) if errors else None,
                   "zone_agreement": sum(zones_ok) / len(zones_ok) if zones_ok else None})
```

(`import math` at the top.) `film/commands/eval.py` prints every key of the result; check it does not hard-code the list.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add film/export/shots.py film/export/evaluate.py tests/test_export_shots.py tests/test_evaluate.py
git commit -m "Export and eval carry the dot: x, y, zone, shot type, and the error against the coach's dot

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 13: Docs, then the first real labels and training (with Garrett)

**Files:**
- Modify: `README.md` (Label court tab, `film prelabel court`, `film train court`, `film court`, the dot keys on the card), `docs/HANDOFF.md` (plan A done; what plan B needs)

- [ ] **Step 1: Document**

README, under "Making it better: label, then train", add item 5:

```
5. **Court.** `uv run film prelabel court <game_id>` queues 40 frames; in **Label court** click each listed point
   where it is in the frame (N when it is not in view), S to save. Then `uv run film train court` (about 30
   minutes) and `uv run film court <game_id>` to fit the mapping through a game. Until plan B lands, dots are
   placed by hand on the shot card: click the half court, arrows nudge a foot, Shift+arrows five.
```

HANDOFF: a dated section listing the new commands, the dataset layout (`datasets/court`, YOLO pose, 18 points, `flip_idx`), `court.parquet`, and that eval's dot error is null until plan B places dots.

- [ ] **Step 2: Commit**

```bash
git add README.md docs/HANDOFF.md
git commit -m "Docs: court labels, the court stage, and dots on the card

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Labels and training (Garrett's go needed)**

Queue 40 frames each from one Hudl game at Marcos de Niza and one iPad game (`film prelabel court <id>`), ask Garrett to label them (about 10 minutes each), then `uv run film train court` (ask first: about 30 minutes on the laptop; `--epochs 30`). Then `uv run film court <id>` on the labelled games (ask first: about 15 minutes each) and report `fitted / frames`, mean `error_px`, and a contact sheet of fitted frames with the template lines drawn back onto them through the inverse mapping, so the fit is checked by eye before plan B builds on it.

---

## Self-review

- Spec coverage: section 2 (court finder) is Tasks 1, 2, 4 to 9 and 13; section 4's zone rules and the folding are Tasks 1 to 3; section 5's card, decisions, queue order and export are Tasks 10 to 12; section 6's eval numbers that exist in plan A are Task 12. People, release, teams and the automatic dot are plans B and C by design.
- Placeholders: none; every code step shows code.
- Type consistency: `Mapping` (Task 2) is what `FakeCourtFinder.mapping` (Task 8) and `load_mappings` (Task 9) return; `POINT_NAMES`/`FLIP_INDEX` (Task 1) are what Tasks 4, 6 and 8 use; `accept_points` (Task 4) is what Task 5's endpoint calls; `zone_of`/`shot_type_of` (Task 3) are what Tasks 10 and 12 call.

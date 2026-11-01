# Court From Reference Pictures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place the court in every frame of a game from 3-5 pictures the coach marks once, instead of 40 marked frames per game and a trained model.

**Architecture:** A camera that only turns (Hudl Focus pans and zooms inside a fixed sensor; an iPad on a tripod turns
on its mount) sees the gym through one homography from any frame to any other. `film/perceive/register.py` finds that
homography between two frames from the floor's own markings (SIFT, ratio test, RANSAC). The coach's marked frames in
`datasets/court` are the **references**. For each sampled frame the references' marked points are carried into the
frame through the registration, and the existing `fit_mapping` turns them into a `Mapping`; a frame no reference lines
up with is carried from the previous frame for up to 30 s. The output is the same `court.parquet` that `film court`
already writes, so nothing downstream changes. A greedy set cover picks which frames to ask the coach to mark.

**Tech Stack:** Python 3.12, OpenCV (SIFT, findHomography), pandas/parquet, FastAPI review app, pytest.

## Global Constraints

- Coordinates in court.parquet, labels and references are full-resolution source pixels; registration works at 960 px wide internally and converts back.
- A reference is a labelled court frame whose id is exactly `<game_id>_<t*10 as 6 digits>`; machine-made training labels end in `a` and are never references.
- The trained-model path (`CourtFinder`) stays; `film court` uses references when the game (or `--refs-from`) has them, else the model.
- Heavy runs on the laptop need Garrett's go (choose references ~3 min, a full pass ~10-20 min).
- Matching is deterministic (brute-force matcher, no random seeds).

---

### Task 1: Frame registration (`film/perceive/register.py`)

**Files:** Create `film/perceive/register.py`; Test `tests/test_register.py`

**Interfaces — Produces:**
- `features(frame_bgr, mask_regions=()) -> Features` (`keypoints (n,2)` full-res px, `descriptors`, `width`, `height`, `scale`)
- `match(a: Features, b: Features) -> Match | None` (`H`: a's px → b's px, `inliers`, `share`)
- `plausible(H, w, h) -> bool`, `project(points: dict, H) -> dict`, `mask_regions(profile) -> list[tuple]`

- [ ] Step 1: tests on a seeded synthetic panorama: two crops with known offsets/zoom recover `H` to < 1.5 px on interior points; unrelated textures give `None`; no keypoint inside a masked region; `plausible` refuses a collapsing or mirrored `H`; `project` maps named points.
- [ ] Step 2: run, see them fail (module missing).
- [ ] Step 3: implement (SIFT 2000 features at 960 px wide, ratio 0.75, RANSAC 3 px at work width, ≥40 inliers and ≥25 % of ratio-test matches, corner quad convex with area ratio in [1/6, 6]; Hudl profiles mask `HUDL_OVERLAY_REGION`).
- [ ] Step 4: tests pass. Step 5: commit.

### Task 2: References and the pass (`film/perceive/court_refs.py`)

**Files:** Create `film/perceive/court_refs.py`; Modify `film/train/datasets.py` (add `read_points`), `film/perceive/court_run.py` (shared row writer, `source` column); Test `tests/test_court_refs.py`, `tests/test_datasets.py`

**Interfaces:**
- Consumes: Task 1; `fit_mapping`, `Mapping`; `ds.write_pending`, `ds.accept_points`.
- Produces: `read_points(dataset, image_id) -> tuple[Path, dict, int, int] | None`; `Reference(image_id, t, points, features)`; `reference_ids(game_id) -> list[str]`; `load_references(game_id, masks) -> list[Reference]`; `place_points(f, refs, prev) -> (points, source, inliers)`; `refs_pass(game_id, refs_from=None, step_s=None, frames=None, train_frames=0, log=print) -> dict`; `check_references(game_id, masks=None) -> list[dict]`.

- [ ] Step 1: tests — `read_points` round-trips `accept_points`; ids ending `a` and frames with no points are not references; on a synthetic panorama with the court drawn through a known feet→pixel map, two references and three frames give sources `ref, ref, chain` and `to_court` within 0.3 ft of truth; a frame past 30 s of chaining is dropped; `train_frames` writes `…a` labels holding only in-frame points; `check_references` reports < 1 px for consistent refs and flags a 20 px slip.
- [ ] Step 2: fail. Step 3: implement. Step 4: pass. Step 5: commit.

### Task 3: Choosing the reference pictures (`choose_references`, `prelabel_court_refs`)

**Files:** Modify `film/perceive/court_refs.py`, `film/train/prelabel.py`, `film/commands/prelabel.py`, `film/review/static/index.html`; Test `tests/test_court_refs.py`, `tests/test_prelabel.py`

**Interfaces — Produces:** `choose_references(game_id, n=90, k_max=5, coverage=0.95, samples=None) -> tuple[list[float], float]`; `prelabel_court_refs(game_id, n=90, k_max=5, log=print) -> int`; `film prelabel court <game> --refs`.

- [ ] Step 1: tests — six overlapping crops of the panorama are covered by ≤ 3 picks at coverage 1.0; an existing reference reduces the picks; queuing writes pending frames with `"ref": true` and removes that game's plain pending court frames only.
- [ ] Step 2-4: fail, implement (greedy set cover over the sample-to-sample match graph, ties to the frame with more links), pass. The Label court tab shows a one-line note on a reference picture.
- [ ] Step 5: commit.

### Task 4: `film court` on references, the contact sheet, docs

**Files:** Create `film/export/court_sheet.py`; Modify `film/commands/court.py`, `README.md`, `docs/HANDOFF.md`; Test `tests/test_court_sheet.py`, `tests/test_cli.py`

**Interfaces — Produces:** `court_lines() -> list[np.ndarray]` (full-court polylines in feet); `draw_court(frame, mapping) -> frame`; `court_sheet(game_id, n=12, log=print) -> Path` (`games/<id>/court_sheet.jpg`); CLI `film court <game> [--refs-from G] [--model] [--train-frames N] [--sheet]`.

- [ ] Step 1: tests — lines drawn through a known mapping land on the expected pixels; the sheet file is written from a court.parquet; the CLI parses the new flags.
- [ ] Step 2-4: fail, implement, pass; full suite green.
- [ ] Step 5: docs (README court section, HANDOFF update) and commit.

### Task 5: On real film (needs Garrett)

- [ ] `film prelabel court <MDN game> --refs` (~3 min, ask first) → Garrett marks the pictures → `film court <game>` → contact sheet checked by eye → `--refs-from` on the other MDN games → same for one iPad game.

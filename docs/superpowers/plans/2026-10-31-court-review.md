# Court Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Garrett approves a court already drawn on each reference picture, and pulls it into place when it is off, instead of marking up to 22 named points. Spec: `docs/superpowers/specs/2026-10-31-court-review-design.md`.

**Architecture:** A draft is the points saved in the pending JSON (`points`, `length`, `draft`). The server does every fit, so the page and `film court` share one set of geometry: the page posts its marks to `/draw` and gets the court's lines back. Approval saves the points in view through the approved fit as an ordinary reference.

**Tech Stack:** Python 3.12, OpenCV, FastAPI, plain JS canvas, pytest.

## Global Constraints

- A pulled spot counts 100 times in the fit (`PIN_WEIGHT = 100`; 10 left a spot pulled 50 px 12 px short).
- Lengths offered are 74, 84, 94 ft. A draft with no length is drawn at 94.
- A carried draft replaces a Claude draft only with 6 or more points; a picture with no draft takes any carried fit.
- A picture with no draft keeps the existing point-by-point walk.
- Labels stay the 22 named points in the existing YOLO pose format.

---

### Task 1: The weighted fit and the court's lines in pixels

**Files:** Modify `film/interpret/court.py`, `film/export/court_sheet.py`; Test `tests/test_court.py`, `tests/test_court_sheet.py`

**Produces:**
- `fit_handles(feet: list[tuple[float, float]], pixels: list[tuple[float, float]], pinned: list[bool], length: float = FULL_DEPTH) -> Mapping | None`
- `court_polylines(mapping: Mapping) -> list[list[list[int]]]`

- [ ] Step 1: tests — a true labelling recovers the template; one pulled spot among slightly wrong marks lands within 3 px of where it was put; left and right swapped returns None; fewer than 4 returns None; `court_polylines` holds a baseline passing through the corners; `draw_court` still draws.
- [ ] Step 2: run, see them fail.
- [ ] Step 3: implement: `_oriented(back, feet)` lifted out of `fit_mapping`, `fit_handles` repeating pinned rows `PIN_WEIGHT` times, `court_polylines` lifted out of `draw_court`.
- [ ] Step 4: pass. Step 5: commit.

### Task 2: Drafts (`film/perceive/court_draft.py`)

**Files:** Create `film/perceive/court_draft.py`, `tests/test_court_draft.py`

**Consumes:** Task 1; `ds.read_pending`, `ds.accept_points`, `ds.list_pending`; `court_refs.load_references`, `place_points`, `points_in_view`, `measure_length`, `REF_ID`.

**Produces:**
- `handle_feet(handle: dict, length: float) -> tuple[float, float]` (a handle is `{"name": str, "px": [x, y], "pinned": bool}` or `{"ft": [x, y], "px": [x, y], "pinned": bool}`)
- `fit_of(handles, length) -> Mapping | None`
- `draw(handles, length) -> dict` (`ok`, `lines`, `H`, `error_px`)
- `approve(image_id, handles, length, seconds=None) -> dict` (`split`, `points`, `carried`; raises `ValueError` on a refused fit)
- `set_draft(image_id, points, length=None, by="claude") -> dict` (`kept`, `dropped`, `error_px`, `picture`; raises `ValueError` on a refused fit)
- `view(image_id, region=None, scale=1.0) -> Path`
- `carry_drafts(game_id) -> int`
- `pending_pictures() -> list[dict]`

- [ ] Step 1: tests — `set_draft` saves points, reports a thrown-out mark, refuses a mirror and writes the check picture; `draw` returns lines and H and `ok: False` on a mirror; `approve` saves every point in view through the fit, leaves the queue, appends the log line with `by`, `pulled`, `moved_px`, `seconds`, and refuses a mirror; on the synthetic panorama `carry_drafts` redraws a picture that lines up and leaves one that does not, and a carried fit under 6 points does not replace a Claude draft.
- [ ] Step 2: fail. Step 3: implement. Step 4: pass. Step 5: commit.

### Task 3: `film court-draft` and the API

**Files:** Create `film/commands/court_draft.py`; Modify `film/cli.py`, `film/commands/prelabel.py`, `film/review/app.py`; Test `tests/test_court_draft.py`, `tests/test_review_app.py`

**Produces:** `film court-draft list | view <id> [--region x0 y0 x1 y1] [--scale s] | set <id> --points JSON [--length L] [--by NAME]`; `POST /api/label/court/{id}/draw`; `POST /api/label/court/{id}/approve`.

- [ ] Step 1: tests — the command saves a draft and prints the kept and dropped marks; `/draw` returns lines, `/approve` empties the queue, a mirrored approve is a 400, an unknown image is a 404.
- [ ] Step 2: fail. Step 3: implement. Step 4: pass. Step 5: commit.

### Task 4: The review mode of Label court

**Files:** Modify `film/review/static/index.html`

- [ ] Step 1: `renderCourtLabel` hands a picture with a draft to `renderCourtReview`: canvas with the court, marks and pulled spots, the keys in the spec, a note when a fit is refused, the seconds on the approve.
- [ ] Step 2: drive it in a browser on a scratch copy of the queue: the court draws, a pull redraws, `L` changes length, `Backspace` and `R` undo, approve advances, a picture with no draft still walks the points.
- [ ] Step 3: run the whole test suite. Step 4: commit.

### Task 5: Draw the queue and hand over

**Files:** Modify `docs/HANDOFF.md`, `README.md`

- [ ] Step 1: draw all 5 queued pictures with `film court-draft`, checking each check picture by eye.
- [ ] Step 2: restart the review server, confirm the first picture shows.
- [ ] Step 3: write how a session draws and what happens after approval into HANDOFF and README. Commit.

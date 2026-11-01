"""A court drawn for the coach to approve, in place of marking points. A draft is the marks saved in a queued picture's
pending file: Claude draws them by eye in a session (`film court-draft`), or they are carried from a picture he has
already approved. He reviews the whole court drawn on the picture, pulls it into place if it is off, and approving
saves it as an ordinary reference (film/perceive/court_refs.py). Spec: docs/superpowers/specs/2026-10-31-court-review-design.md"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

import film.train.datasets as ds
from film.common.manifest import Manifest
from film.common.paths import game_dir, labels_dir
from film.export.court_sheet import court_polylines, draw_court
from film.interpret.court import FULL_DEPTH, Mapping, fit_handles, fit_mapping, template
from film.perceive.court_refs import DATASET, REF_ID, _fit, load_references, measure_length, place_points, points_in_view
from film.perceive.register import features, mask_regions

DRAFT_TOL_PX = 12.0               # further than this from the others' fit, a mark is thrown out of a draft
CARRY_MIN_POINTS = 6              # a carried fit from fewer can be worse than a draft that spans the picture
GRID_MINOR, GRID_MAJOR = 50, 100


def handle_feet(handle: dict, length: float) -> tuple[float, float]:
    """Where on the court a mark is: a named point on a court this long, or the feet read when the coach grabbed it."""
    return template(length)[handle["name"]] if "name" in handle else tuple(handle["ft"])


def fit_of(handles: list[dict], length: float) -> Mapping | None:
    return fit_handles([handle_feet(h, length) for h in handles], [tuple(h["px"]) for h in handles],
                       [bool(h.get("pinned")) for h in handles], length=length)


def draw(handles: list[dict], length: float) -> dict:
    """What the page draws: the court's lines in pixels and the fit, so it can read the feet under the mouse."""
    mapping = fit_of(handles, length)
    if mapping is None:
        return {"ok": False}
    return {"ok": True, "lines": court_polylines(mapping), "H": mapping.H.tolist(), "error_px": round(mapping.error_px, 1)}


def _log(row: dict) -> None:
    path = labels_dir() / DATASET / "review_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _moved_px(handles: list[dict], draft_points: dict) -> float | None:
    """How far the pulled marks ended from where the draft had them. None when he pulled nothing the draft named."""
    moved = [float(np.hypot(h["px"][0] - draft_points[h["name"]][0], h["px"][1] - draft_points[h["name"]][1]))
             for h in handles if h.get("pinned") and h.get("name") in draft_points]
    return round(max(moved), 1) if moved else None


def _masks(game_id: str) -> list:
    return mask_regions(Manifest.load(game_dir(game_id)).profile)


def approve(image_id: str, handles: list[dict], length: float, seconds: float | None = None) -> dict:
    """The court he approved becomes a reference: every template point in view through the fit, so a picture that shows
    both ends or midcourt carries the length he approved. Then the game's other queued pictures are redrawn from it."""
    data = ds.read_pending(DATASET, image_id)
    mapping = fit_of(handles, length)
    if mapping is None:
        raise ValueError("that court is mirrored or has too few marks")
    points = points_in_view(mapping, data["width"], data["height"])
    split = ds.accept_points(DATASET, image_id, points)
    _log({"image_id": image_id, "at": time.time(), "by": (data.get("draft") or {}).get("by"), "length": length,
          "pulled": sum(bool(h.get("pinned")) for h in handles), "moved_px": _moved_px(handles, data.get("points") or {}),
          "seconds": seconds, "points": len(points)})
    found = REF_ID.match(image_id)
    carried = carry_drafts(found.group("game")) if found and (game_dir(found.group("game")) / "manifest.json").exists() else 0
    return {"split": split, "points": len(points), "carried": carried}


def set_draft(image_id: str, points: dict, length: float | None = None, by: str = "claude") -> dict:
    """Save marks as the picture's draft and write a picture of the court they give, to check by eye. A mark far from
    what the others say is left out and named, so it can be looked at again."""
    data = ds.read_pending(DATASET, image_id)
    length = float(length or data.get("length") or FULL_DEPTH)
    given = {n: (float(x), float(y)) for n, (x, y) in points.items()}
    unknown = sorted(set(given) - set(template(length)))
    if unknown:
        raise ValueError(f"unknown points {unknown}")
    mapping = fit_mapping(given, reproj_tol_px=DRAFT_TOL_PX, length=length)
    if mapping is None:
        raise ValueError("those marks give no court: fewer than 4 agree, or left and right are swapped")
    back = np.linalg.inv(mapping.H)
    kept = {}
    for name, (x, y) in given.items():
        v = back @ np.array([*template(length)[name], 1.0])
        if np.hypot(v[0] / v[2] - x, v[1] / v[2] - y) <= DRAFT_TOL_PX * 1.5:
            kept[name] = [x, y]
    data.update(points=kept, length=length, draft={"by": by})
    (ds.pending_dir(DATASET) / f"{image_id}.json").write_text(json.dumps(data, indent=1))
    picture = _drafts_dir() / f"{image_id}.check.jpg"
    out = draw_court(cv2.imread(data["image"]), mapping)
    for x, y in kept.values():
        cv2.drawMarker(out, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.imwrite(str(picture), out)
    return {"kept": sorted(kept), "dropped": sorted(set(given) - set(kept)), "error_px": round(mapping.error_px, 1),
            "length": length, "picture": picture}


def _drafts_dir() -> Path:
    path = labels_dir() / DATASET / "drafts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def view(image_id: str, region: tuple[int, int, int, int] | None = None, scale: float = 1.0) -> Path:
    """A queued picture, or a piece of it, enlarged with a pixel grid numbered in full-picture pixels: what a draft is
    read from."""
    image = cv2.imread(ds.read_pending(DATASET, image_id)["image"])
    h, w = image.shape[:2]
    x0, y0, x1, y1 = region or (0, 0, w, h)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out = cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for gx in range((x0 // GRID_MINOR + 1) * GRID_MINOR, x1, GRID_MINOR):
        at, major = round((gx - x0) * scale), gx % GRID_MAJOR == 0
        cv2.line(out, (at, 0), (at, out.shape[0]), (0, 255, 255) if major else (0, 120, 120), 1)
        if major:
            cv2.putText(out, str(gx), (at + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    for gy in range((y0 // GRID_MINOR + 1) * GRID_MINOR, y1, GRID_MINOR):
        at, major = round((gy - y0) * scale), gy % GRID_MAJOR == 0
        cv2.line(out, (0, at), (out.shape[1], at), (0, 255, 255) if major else (0, 120, 120), 1)
        if major:
            cv2.putText(out, str(gy), (2, at - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    path = _drafts_dir() / f"{image_id}.view.jpg"
    cv2.imwrite(str(path), out)
    return path


def carry_drafts(game_id: str) -> int:
    """Redraw the game's queued pictures from its references wherever they line up. A carried fit from few points does
    not replace a draft that already spans the picture. How many were redrawn."""
    pending = [i for i in ds.list_pending(DATASET) if (m := REF_ID.match(i)) is not None and m.group("game") == game_id]
    if not pending:
        return 0
    masks = _masks(game_id)
    refs = load_references(game_id, masks)
    if not refs:
        return 0
    measured = measure_length(refs)
    count = 0
    for image_id in pending:
        data = ds.read_pending(DATASET, image_id)
        points, _ = place_points(features(cv2.imread(data["image"]), masks), refs)
        w, h = data["width"], data["height"]
        points = {n: p for n, p in points.items() if 0 <= p[0] < w and 0 <= p[1] < h}
        length = float(measured or data.get("length") or FULL_DEPTH)
        if len(points) < (CARRY_MIN_POINTS if data.get("points") else 4) or _fit(points, w, h, length) is None:
            continue
        data.update(points={n: list(p) for n, p in points.items()}, length=length, draft={"by": "carried"})
        (ds.pending_dir(DATASET) / f"{image_id}.json").write_text(json.dumps(data, indent=1))
        count += 1
    return count


def pending_pictures() -> list[dict]:
    """The queued court pictures and who has drawn each so far."""
    out = []
    for image_id in ds.list_pending(DATASET):
        data = ds.read_pending(DATASET, image_id)
        out.append({"image_id": image_id, "by": (data.get("draft") or {}).get("by"), "points": len(data.get("points") or {}),
                    "length": data.get("length")})
    return out

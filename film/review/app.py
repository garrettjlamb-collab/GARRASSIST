from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from film.common.manifest import Manifest
from film.common.paths import game_dir, games_dir
from film.export.clips import clip_path
from film.interpret.events import load_events
from film.perceive.detectors import Box
from film.review.decisions import apply_decisions, load_decisions, save_decision
from film.train import datasets as ds
from film.train.rimclf import CLASSES

STATIC = Path(__file__).parent / "static"


class Decision(BaseModel):
    status: str
    made: bool | None = None
    team: str | None = None


class BoxIn(BaseModel):
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float


class BoxLabel(BaseModel):
    boxes: list[BoxIn]


class ClassLabel(BaseModel):
    cls: str


def _gd(game_id: str) -> Path:
    gd = game_dir(game_id)
    if not (gd / "manifest.json").exists():
        raise HTTPException(404, f"no game {game_id}")
    return gd


def _summary(gd: Path) -> dict:
    m = Manifest.load(gd)
    events = load_events(gd)
    decisions = load_decisions(gd)
    return {"game_id": m.game_id, "profile": m.profile, "date": m.date, "events": len(events),
            "decided": sum(1 for e in events if e.id in decisions),
            "stages": {name: info.get("status") for name, info in m.stages.items()}}


def _safe_summary(gd: Path) -> dict:
    """A game whose manifest or events cannot be read is listed with its error, not allowed to fail the list."""
    try:
        return _summary(gd)
    except Exception as exc:  # noqa: BLE001 - any unreadable file, reported to the page
        return {"game_id": gd.name, "error": f"{type(exc).__name__}: {exc}"}


def _dataset(dataset: str) -> str:
    if dataset not in ds.CLASS_NAMES and dataset != "rimclf":
        raise HTTPException(404, f"no dataset {dataset}")
    return dataset


def _known_image(dataset: str, image_id: str) -> None:
    """404 unless the image is pending or was already accepted (accepts are idempotent)."""
    if image_id in ds.list_pending(dataset):
        return
    accepted = ds._accepted_split(dataset, image_id) if dataset in ds.CLASS_NAMES else (
        ds.find_class_images(dataset, image_id) and "x")
    if not accepted:
        raise HTTPException(404, f"no image {image_id} in {dataset}")


def _range_response(path: Path, range_header: str | None):
    size = path.stat().st_size
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    unsatisfiable = {"Content-Range": f"bytes */{size}"}
    if not match or (not match.group(1) and not match.group(2)):
        raise HTTPException(416, "bad range", headers=unsatisfiable)
    if not match.group(1):
        n = int(match.group(2))
        if n == 0:
            raise HTTPException(416, "bad range", headers=unsatisfiable)
        start, end = max(size - n, 0), size - 1
    else:
        start = int(match.group(1))
        end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
    if start > end or start >= size:
        raise HTTPException(416, "bad range", headers=unsatisfiable)

    def body():
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
    return StreamingResponse(body(), status_code=206, media_type="video/mp4", headers=headers)


def create_app() -> FastAPI:
    app = FastAPI(title="film review")

    @app.get("/api/games")
    def games():
        root = games_dir()
        if not root.exists():
            return []
        return [_safe_summary(p) for p in sorted(root.iterdir()) if (p / "manifest.json").exists()]

    @app.get("/api/games/{game_id}/events")
    def events(game_id: str):
        gd = _gd(game_id)
        return apply_decisions([e.to_json() for e in load_events(gd)], load_decisions(gd))

    @app.post("/api/games/{game_id}/decisions/{event_id}")
    def decide(game_id: str, event_id: str, decision: Decision):
        gd = _gd(game_id)
        if event_id not in {e.id for e in load_events(gd)}:
            raise HTTPException(404, "no such event")
        try:
            return save_decision(gd, event_id, decision.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/clips/{game_id}/{event_id}.mp4")
    def clip(game_id: str, event_id: str):
        path = clip_path(_gd(game_id), event_id)
        if not path.exists():
            raise HTTPException(404, "clip not cut yet; run film clips")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/video/{game_id}")
    def video(game_id: str, request: Request):
        m = Manifest.load(_gd(game_id))
        return _range_response(Path(m.source), request.headers.get("range"))

    @app.get("/api/label/{dataset}/next")
    def label_next(dataset: str):
        _dataset(dataset)
        ids = ds.list_pending(dataset)
        if not ids:
            return {"done": True, "remaining": 0}
        data = ds.read_pending(dataset, ids[0])
        data["remaining"] = len(ids)
        data["image_url"] = f"/labels/{dataset}/{ids[0]}.jpg"
        return data

    @app.get("/labels/{dataset}/{image_id}.jpg")
    def label_image(dataset: str, image_id: str):
        _dataset(dataset)
        path = ds.pending_images_dir(dataset) / f"{image_id}.jpg"
        if not path.exists():
            raise HTTPException(404, "no such pending image")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/label/{dataset}/{image_id}/boxes")
    def label_boxes(dataset: str, image_id: str, body: BoxLabel):
        _dataset(dataset)
        if dataset not in ds.CLASS_NAMES:
            raise HTTPException(400, f"{dataset} is not a box dataset")
        bad = sorted({b.cls for b in body.boxes} - set(ds.CLASS_NAMES[dataset]))
        if bad:
            raise HTTPException(400, f"cls {bad} not in {ds.CLASS_NAMES[dataset]}")
        _known_image(dataset, image_id)
        boxes = [Box(b.x1, b.y1, b.x2, b.y2, 1.0, b.cls) for b in body.boxes]
        try:
            return {"split": ds.accept_boxes(dataset, image_id, boxes)}
        except FileNotFoundError:
            raise HTTPException(404, f"no image {image_id} in {dataset}")

    @app.post("/api/label/{dataset}/{image_id}/class")
    def label_class(dataset: str, image_id: str, body: ClassLabel):
        _dataset(dataset)
        if dataset != "rimclf":
            raise HTTPException(400, f"{dataset} is not a class dataset")
        if body.cls not in CLASSES:
            raise HTTPException(400, f"cls must be one of {CLASSES}")
        _known_image(dataset, image_id)
        try:
            return {"split": ds.accept_class(dataset, image_id, body.cls)}
        except FileNotFoundError:
            raise HTTPException(404, f"no image {image_id} in {dataset}")

    @app.post("/api/label/{dataset}/{image_id}/skip")
    def label_skip(dataset: str, image_id: str):
        _dataset(dataset)
        if image_id not in ds.list_pending(dataset):
            raise HTTPException(404, f"no pending image {image_id} in {dataset}")
        ds.remove_pending(dataset, image_id)
        return {"skipped": image_id}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app

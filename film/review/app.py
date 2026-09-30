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


def _range_response(path: Path, range_header: str | None):
    size = path.stat().st_size
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not match:
        raise HTTPException(416, "bad range")
    start = int(match.group(1) or 0)
    end = min(int(match.group(2) or size - 1), size - 1)
    if start > end:
        raise HTTPException(416, "bad range")

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
        return [_summary(p) for p in sorted(root.iterdir()) if (p / "manifest.json").exists()]

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
            return save_decision(gd, event_id, decision.model_dump(exclude_none=True))
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
        ids = ds.list_pending(dataset)
        if not ids:
            return {"done": True, "remaining": 0}
        data = ds.read_pending(dataset, ids[0])
        data["remaining"] = len(ids)
        data["image_url"] = f"/labels/{dataset}/{ids[0]}.jpg"
        return data

    @app.get("/labels/{dataset}/{image_id}.jpg")
    def label_image(dataset: str, image_id: str):
        path = ds.pending_images_dir(dataset) / f"{image_id}.jpg"
        if not path.exists():
            raise HTTPException(404, "no such pending image")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/label/{dataset}/{image_id}/boxes")
    def label_boxes(dataset: str, image_id: str, body: BoxLabel):
        if dataset not in ds.CLASS_NAMES:
            raise HTTPException(400, f"{dataset} is not a box dataset")
        boxes = [Box(b.x1, b.y1, b.x2, b.y2, 1.0, b.cls) for b in body.boxes]
        return {"split": ds.accept_boxes(dataset, image_id, boxes)}

    @app.post("/api/label/{dataset}/{image_id}/class")
    def label_class(dataset: str, image_id: str, body: ClassLabel):
        if body.cls not in CLASSES:
            raise HTTPException(400, f"cls must be one of {CLASSES}")
        return {"split": ds.accept_class(dataset, image_id, body.cls)}

    @app.post("/api/label/{dataset}/{image_id}/skip")
    def label_skip(dataset: str, image_id: str):
        ds.remove_pending(dataset, image_id)
        return {"skipped": image_id}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app

"""`film statbook`: put the stat book's plays on the film, write the play list, cut labeled clips and reels."""
from __future__ import annotations

import csv
import subprocess
import json
import re
from pathlib import Path

from film.common.cloud import may_need_film
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.ffmpeg import ffmpeg_exe
from film.export.clips import build_reel, cut_clip
from film.interpret.events import load_events
from film.interpret.quarters import load_quarters
from film.interpret.statbook import Placed, cards_from_events, label, load_statbook, place_game
from film.perceive.scoreboard import load_timeline, lookup
from film.review.decisions import load_decisions

def shot_sides(gd: Path, events, frame_w: int) -> dict:
    """Each card's basket from where its rim sits in the frame at the moment of the shot. The panning camera
    frames the basket being attacked toward its own end: at Marcos de Niza the PADRES-end rim sits about 37%
    across, the other about 63%. (A rim track's median over the game, `basket_sides`, mixes the two when the
    tracker hands one id across both ends.) None when the rim is near the middle."""
    from film.perceive.tracks import read_tracks

    rims = read_tracks(gd, "rim")
    if rims.empty:
        return {}
    out = {}
    for e in events:
        d = rims[(rims["track_id"] == e.rim_id) & ((rims["t"] - e.t).abs() < 0.6)]
        if d.empty:
            continue
        cx = float(((d["x1"] + d["x2"]) / 2).median()) / frame_w
        if abs(cx - 0.5) >= SIDE_MARGIN:
            out[e.id] = "left" if cx < 0.5 else "right"
    return out


SIDE_MARGIN = 0.05
CLIP_TYPES = ("shot", "reb", "ast", "stl", "blk", "to", "foul")
RESULT_FILE = "statbook.json"


def _clock(s: float | None) -> str:
    return "--" if s is None else f"{int(s // 60)}:{int(s % 60):02d}"


def _hms(t: float | None) -> str:
    if t is None:
        return ""
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def match_game(game_id: str, statbook_path: Path) -> dict:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    book = load_statbook(statbook_path)
    events = load_events(gd)
    rejected = {k for k, v in load_decisions(gd).items() if v.get("status") == "rejected"}
    sides = shot_sides(gd, events, m.width)
    timeline = load_timeline(gd / "scoreboard.jsonl")

    readable = sum(r.clock_s is not None for r in timeline) / max(1, len(timeline))

    def clock_at(t: float):
        if readable < 0.3:                        # a board the reader cannot follow: no clock beats a wrong one
            return None
        r = lookup(timeline, t)
        return None if r is None or r.clock_s is None or t - r.t > 3 else r.clock_s

    marks = load_quarters(gd)                     # the coach's quarter marks win over the scoreboard
    spans = {k: span for k, span in enumerate(marks, 1)} if marks else None
    placed, periods = place_game(book, cards_from_events(events, sides, rejected), timeline, clock_at, film_spans=spans)
    them = book.game.get("them") or "Opp"
    plays = []
    for x in placed:
        d = x.to_json()
        d["label"] = label(x, book.roster, them)
        plays.append(d)
    result = {"format": "film-statbook", "version": 1, "game_id": game_id, "statbook": str(statbook_path),
              "game": book.game, "roster": book.roster,
              "periods": [p.__dict__ for p in periods], "plays": plays}
    (gd / RESULT_FILE).write_text(json.dumps(result, indent=1))
    out = gd / "export"
    out.mkdir(exist_ok=True)
    with (out / f"{game_id}.plays.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "period", "film_time", "film_t", "clock", "label", "type", "team", "player", "name",
                    "made", "pts", "how", "uncertainty_s", "card_id", "score_us", "score_them"])
        for p in plays:
            w.writerow([p["seq"], p["period"], _hms(p["film_t"]), p["film_t"], _clock(p["clock_s"]), p["label"], p["type"],
                        p["team"], p["player"], p["name"], p["made"], p["pts"], p["how"], p["uncertainty_s"],
                        p["card_id"], p["score_us"], p["score_them"]])
    return result


def load_result(game_id: str) -> dict:
    return json.loads((game_dir(game_id) / RESULT_FILE).read_text())


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def select(plays: list[dict], player: str | None = None, types: list[str] | None = None,
           team: str | None = None) -> list[dict]:
    out = []
    for p in plays:
        if p.get("film_t") is None or p["type"] not in CLIP_TYPES:
            continue
        if types and p["type"] not in types:
            continue
        if team and p.get("team") != team:
            continue
        if player:
            key = player.lower()
            if key not in {str(p.get("player") or "").lower(), str(p.get("name") or "").lower()}:
                continue
        out.append(p)
    return out


def clip_file(gd: Path, p: dict) -> Path:
    return gd / "statbook-clips" / f"{p['seq']:03d}-Q{p['period']}-{_slug(p['label'])}.mp4"


@may_need_film
def cut_clips(game_id: str, plays: list[dict], force: bool = False, log=print, *, film) -> list[Path]:
    gd = game_dir(game_id)
    paths = []
    for p in plays:
        out = clip_file(gd, p)
        if force or not out.exists():
            cut_clip(film(), p["clip"]["start"], p["clip"]["end"], out)
        paths.append(out)
    log(f"statbook clips {game_id}: {len(paths)} ready in {gd / 'statbook-clips'}")
    return paths


def reel(game_id: str, plays: list[dict], name: str, log=print) -> Path:
    gd = game_dir(game_id)
    paths = cut_clips(game_id, plays, log=log)
    labels = [f"Q{p['period']} {_clock(p.get('clock_s'))}  {p['label']}" for p in plays]
    out = gd / "export" / f"{game_id}.{_slug(name)}.reel.mp4"
    out.parent.mkdir(exist_ok=True)
    raw = build_reel(list(zip(paths, labels)), out.with_suffix(".raw.mp4"))
    # OpenCV writes mp4v at about 0.4 MB a second; H.264 is a seventh of that and plays on a phone
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    return out

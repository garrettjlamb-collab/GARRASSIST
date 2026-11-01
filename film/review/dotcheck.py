"""The coach's own dots on a few live shots, placed without seeing the machine's, so the machine's can be measured in feet. The
page gets no machine dot; each answer is logged with the machine's dot as it stood, and goes on the shot's decision as his dot."""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

from film.common.manifest import Manifest
from film.common.paths import game_dir, games_dir, labels_dir
from film.export.clips import clip_path
from film.interpret.court import BASKET, THREE_RADIUS
from film.interpret.events import load_events
from film.interpret.zones import is_three, zone_of
from film.review.decisions import load_decisions, save_decision

N_DEFAULT = 30
CLOSE_FT = 9.0
SEED = 7


def _dir() -> Path:
    path = labels_dir() / "dotcheck"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_queue() -> list[dict]:
    path = _dir() / "queue.json"
    return json.loads(path.read_text()) if path.exists() else []


def read_answers() -> list[dict]:
    path = _dir() / "answers.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _kind(x: float, y: float) -> str:
    dist = math.hypot(x - BASKET[0], y - BASKET[1])
    return "three" if dist > THREE_RADIUS else "close" if dist <= CLOSE_FT else "mid"


def _candidates(gd: Path) -> list:
    decisions = load_decisions(gd)
    out = []
    for e in load_events(gd):
        d = decisions.get(e.id) or {}
        if d.get("status") != "confirmed" or d.get("x") is not None or e.signals.get("phase") == "warmup" or e.x is None:
            continue
        if (e.signals.get("location") or {}).get("method") not in ("flight", "path", "pose") or not clip_path(gd, e.id).exists():
            continue
        out.append(e)
    return out


def pick(n: int = N_DEFAULT, seed: int = SEED, log=print) -> list[dict]:
    """Shots spread over every game with dots and over the machine's three kinds of shot, in mixed order. Free throws are left out:
    they are on the line by rule, so a dot there measures nothing."""
    rng = random.Random(seed)
    games = sorted(p for p in games_dir().iterdir() if (p / "manifest.json").exists())
    pools = {}
    for gd in games:
        found = _candidates(gd)
        if found:
            rng.shuffle(found)
            pools[gd.name] = found
    if not pools:
        return []
    chosen: list[dict] = []
    # Round by round: each game gives its next kind of shot in turn, so every game and kind is covered before any repeats.
    order = ["three", "mid", "close"]
    round_ = 0
    while len(chosen) < n and any(pools.values()):
        kind = order[round_ % len(order)]
        for name, found in pools.items():
            if len(chosen) >= n or not found:
                continue
            e = next((x for x in found if _kind(x.x, x.y) == kind), found[0])
            found.remove(e)
            chosen.append({"game_id": name, "event_id": e.id})
        round_ += 1
    rng.shuffle(chosen)
    (_dir() / "queue.json").write_text(json.dumps(chosen, indent=1))
    log(f"dot check: {len(chosen)} shots queued from {len(pools)} games")
    return chosen


def _done() -> set[tuple[str, str]]:
    return {(a["game_id"], a["event_id"]) for a in read_answers()}


def next_shot() -> dict:
    """The next shot for the page, with nothing on it that says where the machine put the dot."""
    queue, done = read_queue(), _done()
    left = [q for q in queue if (q["game_id"], q["event_id"]) not in done]
    if not left:
        return {"done": True, "total": len(queue), "answered": len(done), "remaining": 0}
    q = left[0]
    event = next((e for e in load_events(game_dir(q["game_id"])) if e.id == q["event_id"]), None)
    if event is None:
        raise KeyError(q["event_id"])
    return {"done": False, "total": len(queue), "answered": len(queue) - len(left), "remaining": len(left), "game_id": q["game_id"], "event_id": q["event_id"],
            "title": Manifest.load(game_dir(q["game_id"])).title, "t": event.t, "clip": event.clip, "rim_id": event.rim_id, "made": event.made}


def _log(row: dict) -> None:
    with (_dir() / "answers.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def _queued(game_id: str, event_id: str) -> bool:
    return any(q["game_id"] == game_id and q["event_id"] == event_id for q in read_queue())


def answer(game_id: str, event_id: str, x: float, y: float) -> dict:
    """His dot: logged beside the machine's as it stands now, and saved on the shot's decision."""
    if not _queued(game_id, event_id):
        raise KeyError(event_id)
    gd = game_dir(game_id)
    event = next(e for e in load_events(gd) if e.id == event_id)
    previous = load_decisions(gd).get(event_id) or {}
    save_decision(gd, event_id, {**{k: previous[k] for k in ("status", "made", "team", "foul") if k in previous}, "x": x, "y": y})
    loc = event.signals.get("location") or {}
    _log({"game_id": game_id, "event_id": event_id, "at": time.time(), "x": x, "y": y, "machine_x": event.x, "machine_y": event.y, "method": loc.get("method"),
          "conf": event.location_conf, "flags": event.flags, "profile": Manifest.load(gd).profile, "made": event.made, "points": event.signals.get("points")})
    return next_shot()


def skip(game_id: str, event_id: str) -> dict:
    """He cannot tell where it was taken from. Nothing goes on the shot."""
    if not _queued(game_id, event_id):
        raise KeyError(event_id)
    _log({"game_id": game_id, "event_id": event_id, "at": time.time(), "skipped": True})
    return next_shot()


def mark_doubtful(shots: list[int]) -> list[dict]:
    """Answers he says may be for the wrong shot, by their number on the page. They leave the score and his dot comes off the shot,
    so it cannot replace the machine's in the export; the answer stays in the log, marked."""
    answers = read_answers()
    marked = []
    for n in shots:
        a = answers[n - 1]
        a["doubtful"] = True
        marked.append(a)
        if a.get("skipped"):
            continue
        gd = game_dir(a["game_id"])
        previous = load_decisions(gd).get(a["event_id"]) or {}
        save_decision(gd, a["event_id"], {**{k: previous[k] for k in ("status", "made", "team", "foul") if k in previous}, "x": None, "y": None})
    tmp = _dir() / "answers.tmp"
    tmp.write_text("".join(json.dumps(a) + "\n" for a in answers))
    tmp.replace(_dir() / "answers.jsonl")
    return marked


def score(answers: list[dict] | None = None) -> dict:
    answers = read_answers() if answers is None else answers
    doubtful = sum(bool(a.get("doubtful")) for a in answers)
    rows = [a for a in answers if not a.get("skipped") and not a.get("doubtful") and a.get("machine_x") is not None]
    if not rows:
        return {"answered": 0, "skipped": sum(bool(a.get("skipped")) for a in answers), "doubtful": doubtful}

    def summary(group: list[dict]) -> dict:
        err = [math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"]) for a in group]
        return {"n": len(group), "median_ft": round(median(err), 1), "within_3ft": round(sum(e <= 3 for e in err) / len(err), 2), "within_6ft": round(sum(e <= 6 for e in err) / len(err), 2),
                "over_10ft": round(sum(e > 10 for e in err) / len(err), 2),
                "arc_side": round(sum(is_three(a["x"], a["y"]) == is_three(a["machine_x"], a["machine_y"]) for a in group) / len(group), 2),
                "same_zone": round(sum(zone_of(a["x"], a["y"]) == zone_of(a["machine_x"], a["machine_y"]) for a in group) / len(group), 2)}

    by = defaultdict(list)
    for a in rows:
        by["camera " + a["profile"]].append(a)
        by["method " + str(a["method"])].append(a)
        by["his shot " + _kind(a["x"], a["y"])].append(a)
    # The same shots with the machine's dot mirrored: much better would mean the two courts disagree on which side is left.
    mirrored = median(math.hypot(a["x"] - (50 - a["machine_x"]), a["y"] - a["machine_y"]) for a in rows)
    return {"answered": len(rows), "skipped": sum(bool(a.get("skipped")) for a in answers), "doubtful": doubtful, "all": summary(rows), "mirrored_median_ft": round(mirrored, 1),
            **{k: summary(v) for k, v in sorted(by.items())}}

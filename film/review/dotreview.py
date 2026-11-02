"""The machine's dots put in front of the coach, worst first. About one card in ten hides the machine's dot: his click there is independent
of it, so those cards keep measuring the dots, where an approval made while looking at the dot could not. Every dot saved says how it got there
(`dot_source`): coach (placed or moved by him), approved (he looked and kept it), bulk (kept unseen)."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from statistics import median

from film.common.manifest import Manifest
from film.common.paths import game_dir, games_dir, labels_dir
from film.export.clips import clip_path
from film.interpret.events import load_events
from film.interpret.zones import is_three, zone_of
from film.review.decisions import load_decisions, save_decision

HIGH_CONF = 0.8
BLIND_ONE_IN = 10
TRUST_CHECK_ONE_IN = 5
SAME_DOT_FT = 0.05
KEEP = ("status", "made", "team", "foul")


def _log_path() -> Path:
    path = labels_dir() / "dotreview"
    path.mkdir(parents=True, exist_ok=True)
    return path / "answers.jsonl"


def read_answers() -> list[dict]:
    path = _log_path()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _log(row: dict) -> None:
    with _log_path().open("a") as f:
        f.write(json.dumps({**row, "at": time.time()}) + "\n")


def is_blind(game_id: str, event_id: str) -> bool:
    """Fixed for a shot, however the queue shrinks."""
    return int(hashlib.sha1(f"{game_id}/{event_id}".encode()).hexdigest(), 16) % BLIND_ONE_IN == 0


def is_trust_check(game_id: str, event_id: str) -> bool:
    """One in five of the dots the trust model vouches for stays hidden, because its claim was chosen on the answers it was tested on."""
    return int(hashlib.sha1(f"trust/{game_id}/{event_id}".encode()).hexdigest(), 16) % TRUST_CHECK_ONE_IN == 0


def _rows() -> list[dict]:
    """Every confirmed live shot with no dot of his yet and nothing he skipped. Free throws are on the line by rule and are left out."""
    skipped = {(a["game_id"], a["event_id"]) for a in read_answers() if a.get("skipped")}
    rows = []
    for gd in sorted(p for p in games_dir().iterdir() if (p / "events.json").exists() and (p / "manifest.json").exists()):   # a game with no court has no machine dots: he places them all
        decisions = load_decisions(gd)
        for e in load_events(gd):
            d = decisions.get(e.id) or {}
            loc = e.signals.get("location") or {}
            if d.get("status") != "confirmed" or d.get("x") is not None or e.signals.get("phase") == "warmup" or loc.get("method") == "free-throw":
                continue
            if (gd.name, e.id) in skipped or not clip_path(gd, e.id).exists():
                continue
            rows.append({"game_id": gd.name, "event_id": e.id, "x": e.x, "y": e.y, "conf": e.location_conf, "method": loc.get("method"), "made": d.get("made", e.made),
                         "blind": is_blind(gd.name, e.id), "t": e.t, "clip": e.clip, "rim_id": e.rim_id, "trust": e.signals.get("trust")})
            if rows[-1]["trust"] and rows[-1]["trust"]["trusted"] and is_trust_check(gd.name, e.id):
                rows[-1]["blind"] = True
    return rows


def _sure(r: dict) -> bool:
    """Once the trust model has judged a shot its word stands; before that, the machine's own confidence."""
    return r["trust"]["trusted"] if r.get("trust") else (r["conf"] or 0) >= HIGH_CONF


def _band(r: dict) -> int:
    """0 sure dots hidden to check them, first: they say whether keeping the sure dots unseen is safe. 1 the doubtful dots,
    2 no dot at all, 3 sure dots that can be kept in one go."""
    if r["x"] is None:
        return 2
    if not _sure(r):
        return 1
    return 0 if r["blind"] else 3


def queue() -> list[dict]:
    return sorted(_rows(), key=lambda r: (_band(r), (r["trust"] or {}).get("prob", r["conf"] or 0), r["game_id"], r["t"]))


def next_shot() -> dict:
    rows = queue()
    counts = {"remaining": len(rows), "doubtful": sum(_band(r) == 1 for r in rows), "no_dot": sum(_band(r) == 2 for r in rows), "bulk": sum(_band(r) == 3 for r in rows),
              "checks": sum(_band(r) == 0 for r in rows), "done_so_far": sum(not a.get("skipped") for a in read_answers())}
    if not rows:
        return {"done": True, **counts}
    r = rows[0]
    card = {"done": False, **counts, "game_id": r["game_id"], "event_id": r["event_id"], "title": Manifest.load(game_dir(r["game_id"])).title, "t": r["t"], "clip": r["clip"],
            "rim_id": r["rim_id"], "made": r["made"], "blind": r["blind"] or r["x"] is None}
    if not card["blind"]:                  # a hidden card carries nothing that says where the machine put the dot
        card.update(x=r["x"], y=r["y"], conf=(r["trust"] or {}).get("prob", r["conf"]))
    return card


def _find(game_id: str, event_id: str) -> dict:
    found = next((r for r in _rows() if r["game_id"] == game_id and r["event_id"] == event_id), None)
    if found is None:
        raise KeyError(event_id)
    return found


def _save(r: dict, x: float, y: float, source: str) -> None:
    gd = game_dir(r["game_id"])
    previous = load_decisions(gd).get(r["event_id"]) or {}
    save_decision(gd, r["event_id"], {**{k: previous[k] for k in KEEP if k in previous}, "x": x, "y": y, "dot_source": source})


def answer(game_id: str, event_id: str, x: float | None = None, y: float | None = None) -> dict:
    """A dot from him, or with no dot given, the machine's kept."""
    r = _find(game_id, event_id)
    blind = r["blind"] or r["x"] is None
    if x is None or y is None:
        if blind:
            raise ValueError("this card needs a dot")
        x, y, source = r["x"], r["y"], "approved"
    else:
        same = r["x"] is not None and math.hypot(x - r["x"], y - r["y"]) <= SAME_DOT_FT and not blind
        source = "approved" if same else "coach"
    _save(r, x, y, source)
    _log({"game_id": game_id, "event_id": event_id, "x": x, "y": y, "machine_x": r["x"], "machine_y": r["y"], "conf": r["conf"], "method": r["method"], "blind": blind, "source": source, "trust": r["trust"]})
    return next_shot()


def skip(game_id: str, event_id: str) -> dict:
    _find(game_id, event_id)
    _log({"game_id": game_id, "event_id": event_id, "skipped": True})
    return next_shot()


def no_shot(game_id: str, event_id: str) -> dict:
    """He watched it and there is no shot in the clip: it stops being a shot. Different from skip, which only says he cannot place it."""
    _find(game_id, event_id)
    gd = game_dir(game_id)
    previous = load_decisions(gd).get(event_id) or {}
    save_decision(gd, event_id, {**{k: previous[k] for k in ("made", "team", "foul") if k in previous}, "status": "rejected"})
    _log({"game_id": game_id, "event_id": event_id, "skipped": True, "no_shot": True})
    return next_shot()


def bulk_approve(log=print) -> int:
    """Keep the machine's sure dots unseen. Cards hidden to check them stay in the queue."""
    rows = [r for r in queue() if _band(r) == 3]
    for r in rows:
        _save(r, r["x"], r["y"], "bulk")
        _log({"game_id": r["game_id"], "event_id": r["event_id"], "x": r["x"], "y": r["y"], "machine_x": r["x"], "machine_y": r["y"], "conf": r["conf"], "method": r["method"], "blind": False, "source": "bulk", "trust": r["trust"]})
    log(f"dot review: {len(rows)} dots kept unseen")
    return len(rows)


def score(answers: list[dict] | None = None) -> dict:
    """Hidden cards measure the dots. Shown cards only say how often he changed one, and he was looking at it when he chose."""
    answers = read_answers() if answers is None else answers
    rows = [a for a in answers if not a.get("skipped")]

    def measured(group: list[dict]) -> dict | None:
        group = [a for a in group if a["machine_x"] is not None]
        if not group:
            return None
        err = [math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"]) for a in group]
        return {"n": len(group), "median_ft": round(median(err), 1), "within_3ft": round(sum(e <= 3 for e in err) / len(err), 2), "over_10ft": round(sum(e > 10 for e in err) / len(err), 2),
                "arc_side": round(sum(is_three(a["x"], a["y"]) == is_three(a["machine_x"], a["machine_y"]) for a in group) / len(group), 2),
                "same_zone": round(sum(zone_of(a["x"], a["y"]) == zone_of(a["machine_x"], a["machine_y"]) for a in group) / len(group), 2)}

    blind = [a for a in rows if a.get("blind")]
    shown = [a for a in rows if not a.get("blind") and a["source"] != "bulk"]
    high = lambda a: (a.get("conf") or 0) >= HIGH_CONF
    judged = [a for a in blind if a.get("trust")]
    return {"hidden_since_trust": {"trusted": measured([a for a in judged if a["trust"]["trusted"]]), "not_trusted": measured([a for a in judged if not a["trust"]["trusted"]])},
            "hidden": {"all": measured(blind), "conf_0.8_up": measured([a for a in blind if high(a)]), "conf_under_0.8": measured([a for a in blind if not high(a)])},
            "shown_not_independent": {"approved": sum(a["source"] == "approved" for a in shown), "moved": sum(a["source"] == "coach" for a in shown), "moved_by": measured([a for a in shown if a["source"] == "coach"])},
            "bulk": sum(a["source"] == "bulk" for a in rows), "skipped": sum(bool(a.get("skipped")) for a in answers), "no_machine_dot_placed": sum(a.get("machine_x") is None for a in rows)}

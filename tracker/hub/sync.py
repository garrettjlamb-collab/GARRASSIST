"""Copy stat book games into the season hub's store, slimmed: the hub never reads the stat book itself.

1. Claude reads the stat book's store (ArtifactData list games -> tracker/backups/<date>/games/*.json).
2. python3 tracker/hub/sync.py tracker/backups/<date>/games  -> tracker/hub/out/games/<id>.json
3. Claude writes each file to the hub's store (ArtifactData batch set games/<id>, file_path).

Read-only on the stat book's data: the backup files are only read. Games whose name says "test" are
left out (pass --all to keep them).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

GYM_TZ = ZoneInfo("America/Phoenix")   # games are played in Arizona; the stat book once stamped UTC dates

KEEP = ("id", "type", "team", "player", "made", "pts", "off", "p", "on", "in", "out", "x", "y", "ref", "kind",
        "action", "period", "ms", "def")


KINDS = Path(__file__).parent / "kinds.json"       # game id -> "scrimmage": kept in the hub, never in season totals
# game id -> event id -> fields to change, so the hub matches the official book. The stat book is never edited.
CORRECTIONS = Path(__file__).parent / "corrections.json"


def local_date(game: dict) -> str | None:
    """The gym's date for a game: from its creation time in Arizona, else whatever the stat book wrote."""
    created = game.get("created")
    if created:
        try:
            return datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(GYM_TZ).date().isoformat()
        except ValueError:
            pass
    return game.get("date")


def slim(game: dict, kinds: dict | None = None, fixes: dict | None = None) -> dict:
    fix = (fixes or {}).get(game["id"], {})
    # A play the stat book never logged: {"add:<new id>": {"after": <event id>, ...the play's fields, "why"}}
    adds: dict[str, list[dict]] = {}
    for key, f in fix.items():
        if key.startswith("add:"):
            play = {k: v for k, v in f.items() if k not in ("why", "after")}
            adds.setdefault(f["after"], []).append(dict(play, id=key[4:]))
    missing = set(adds) - {e.get("id") for e in game["events"]}
    if missing:
        raise SystemExit(f"{game['id']}: a correction adds a play after {sorted(missing)}, which is not in the game")
    events = []
    for e in game["events"]:
        e = dict(e, **{k: v for k, v in fix.get(e.get("id"), {}).items() if k != "why"})
        events.append({k: e[k] for k in KEEP if k in e and e[k] is not None})
        for play in adds.get(e.get("id"), []):
            # the added play happens with the same five on the floor and in the same period as the one before it
            play = {"on": e.get("on"), "p": e.get("p"), "ms": e.get("ms"), **play}
            events.append({k: play[k] for k in KEEP if k in play and play[k] is not None})
    return {"id": game["id"], "date": local_date(game), "us": game.get("us"), "them": game.get("them"),
            "qlen": game.get("qlen"), "roster": game.get("roster", []), "starters": game.get("starters", []),
            "updated": game.get("updated"), "kind": (kinds or {}).get(game["id"], "game"), "events": events}


def is_test(game: dict) -> bool:
    words = " ".join(str(game.get(k) or "") for k in ("id", "us", "them")).lower()
    return "test" in words


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("games_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "games")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    kinds = json.loads(KINDS.read_text()) if KINDS.exists() else {}
    fixes = json.loads(CORRECTIONS.read_text()) if CORRECTIONS.exists() else {}
    for path in sorted(args.games_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        game = doc.get("data", doc)
        if is_test(game) and not args.all:
            print(f"skip  {game['id']} (test game)")
            continue
        out = args.out / f"{game['id']}.json"
        text = json.dumps(slim(game, kinds, fixes), separators=(",", ":"))
        if len(text) > 250_000:
            print(f"error {game['id']}: {len(text)} bytes is over the store's 256 KiB document limit", file=sys.stderr)
            return 1
        out.write_text(text)
        n_fix = len(fixes.get(game["id"], {}))
        print(f"wrote {out}  {len(text)} bytes, {len(game['events'])} plays, {kinds.get(game['id'], 'game')}"
              + (f", {n_fix} correction(s) to match the official book" if n_fix else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

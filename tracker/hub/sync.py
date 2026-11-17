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
from pathlib import Path

KEEP = ("id", "type", "team", "player", "made", "pts", "off", "p", "on", "in", "out", "x", "y", "ref", "kind",
        "action", "period", "ms")


KINDS = Path(__file__).parent / "kinds.json"       # game id -> "scrimmage": kept in the hub, never in season totals


def slim(game: dict, kinds: dict | None = None) -> dict:
    events = []
    for e in game["events"]:
        events.append({k: e[k] for k in KEEP if k in e and e[k] is not None})
    return {"id": game["id"], "date": game.get("date"), "us": game.get("us"), "them": game.get("them"),
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
    for path in sorted(args.games_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        game = doc.get("data", doc)
        if is_test(game) and not args.all:
            print(f"skip  {game['id']} (test game)")
            continue
        out = args.out / f"{game['id']}.json"
        text = json.dumps(slim(game, kinds), separators=(",", ":"))
        if len(text) > 250_000:
            print(f"error {game['id']}: {len(text)} bytes is over the store's 256 KiB document limit", file=sys.stderr)
            return 1
        out.write_text(text)
        print(f"wrote {out}  {len(text)} bytes, {len(game['events'])} plays, {kinds.get(game['id'], 'game')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

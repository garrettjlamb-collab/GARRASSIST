"""The three low-camera games after film locate: dots, kinds of dot, how the trust model sorts them, and the scoreboard where one reads."""
import json
import math
from collections import Counter
from pathlib import Path

from film.interpret.events import Event
from film.interpret.zones import is_three
from film.perceive.shotspots import confirmed_shots
from film.review.decisions import load_decisions

for g in ("2026-10-01-combs-vs-mdn-angle-1", "2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1", "2026-10-09-new-cactus-shadows-angle-1"):
    gd = Path("games") / g
    shots = confirmed_shots(gd)
    m = Counter((e.signals.get("location") or {}).get("method") for e in shots)
    t = Counter()
    for e in shots:
        tr = e.signals.get("trust")
        if e.x is None:
            t["no dot"] += 1
        elif (e.signals.get("location") or {}).get("method") == "free-throw":
            t["free throw"] += 1
        elif not tr:
            t["not judged"] += 1
        else:
            t["trusted" if tr["trusted"] else "doubtful"] += 1
    made = [e for e in shots if e.x is not None and e.made and e.signals.get("points") in (2, 3) and e.signals["location"]["method"] != "free-throw"]
    r = sum(is_three(e.x, e.y) == (e.signals["points"] == 3) for e in made)
    print(g[11:40], "shots", len(shots), dict(m), dict(t), "| scoreboard side", f"{r}/{len(made)}")
    before = load_decisions(gd)
    old = json.loads((gd / "review.before-locate.json").read_text())
    print("   decisions unchanged:", before == old)

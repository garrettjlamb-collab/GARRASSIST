"""The coach's dots against the machine's as they stand on disk now, beside what they were when he placed his. Run once per change of rules:
a change tuned on this result needs a fresh batch of his dots to be measured."""
import json
import math
from pathlib import Path
from statistics import median

from film.common.paths import game_dir
from film.interpret.events import load_events
from film.interpret.zones import is_three, zone_of
from film.review.dotcheck import read_answers

rows = [a for a in read_answers() if not a.get("skipped") and not a.get("doubtful")]
for a in rows:
    e = next(x for x in load_events(game_dir(a["game_id"])) if x.id == a["event_id"])
    a["now_x"], a["now_y"], a["now_method"], a["now_conf"] = e.x, e.y, (e.signals.get("location") or {}).get("method"), e.location_conf


def summary(name, g, xk, yk):
    g = [a for a in g if a[xk] is not None]
    if not g:
        return
    err = [math.hypot(a["x"] - a[xk], a["y"] - a[yk]) for a in g]
    print(f"{name:26s} n{len(g):3d} median {median(err):5.1f} ft | within 3 {sum(v <= 3 for v in err) / len(err):4.0%} within 6 {sum(v <= 6 for v in err) / len(err):4.0%} over 10 {sum(v > 10 for v in err):2d} ({sum(v > 10 for v in err) / len(err):3.0%}) | arc {sum(is_three(a['x'], a['y']) == is_three(a[xk], a[yk]) for a in g) / len(g):4.0%} zone {sum(zone_of(a['x'], a['y']) == zone_of(a[xk], a[yk]) for a in g) / len(g):4.0%}")


summary("when he placed his", rows, "machine_x", "machine_y")
summary("now", rows, "now_x", "now_y")
for p in sorted({a["profile"] for a in rows}):
    summary("  camera " + p, [a for a in rows if a["profile"] == p], "now_x", "now_y")
for m in ("pose", "path", "flight"):
    summary("  method " + m, [a for a in rows if a["now_method"] == m], "now_x", "now_y")
summary("  conf 0.8 and up", [a for a in rows if (a["now_conf"] or 0) >= 0.8], "now_x", "now_y")
summary("  conf under 0.8", [a for a in rows if (a["now_conf"] or 0) < 0.8], "now_x", "now_y")
print("moved over 2 ft:", sum(math.hypot(a["machine_x"] - a["now_x"], a["machine_y"] - a["now_y"]) > 2 for a in rows if a["now_x"] is not None))
for i, a in enumerate(rows):
    if a["now_x"] is None:
        continue
    before, after = math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"]), math.hypot(a["x"] - a["now_x"], a["y"] - a["now_y"])
    if before > 8 or after > 8:
        print(f"  {a['game_id'][11:24]} {a['event_id']:18s} his ({a['x']:.0f},{a['y']:.0f}) was {before:4.1f} now {after:4.1f} {a['now_method']} conf {a['now_conf']}")

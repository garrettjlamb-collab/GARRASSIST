"""What the coach's dot review says: time spent, hidden cards by confidence, moved rates by method and camera, and his dots at the free-throw line."""
import json
import math
from collections import Counter, defaultdict
from statistics import median

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.court import FT_LINE_Y, WIDTH
from film.review.dotreview import read_answers

rows = read_answers()
done = [a for a in rows if not a.get("skipped")]
print("answers", len(rows), "skipped", sum(bool(a.get("skipped")) for a in rows), Counter(a.get("source") for a in done))
ts = sorted(a["at"] for a in rows if a.get("source") != "bulk")
gaps = [b - a for a, b in zip(ts, ts[1:])]
print("median s per shot", round(median(gaps), 1), "active minutes", round(sum(g for g in gaps if g < 120) / 60, 1))
prof = {}
for a in rows:
    prof.setdefault(a["game_id"], Manifest.load(game_dir(a["game_id"])).profile)


def err(a):
    return math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"])


def line(name, g):
    g = [a for a in g if a.get("machine_x") is not None]
    if not g:
        return
    e = [err(a) for a in g]
    print(f"  {name:30s} n{len(g):3d} median {median(e):5.1f} within3 {sum(v <= 3 for v in e) / len(e):4.0%} over10 {sum(v > 10 for v in e) / len(e):4.0%}")


blind = [a for a in done if a.get("blind")]
shown = [a for a in done if not a.get("blind")]
print("HIDDEN cards")
line("all", blind)
for lo, hi in ((0, .5), (.5, .8), (.8, 1.01)):
    line(f"conf {lo}-{hi}", [a for a in blind if lo <= (a["conf"] or 0) < hi])
for m in ("pose", "path", "flight"):
    line("method " + m, [a for a in blind if a["method"] == m])
for p in sorted(set(prof.values())):
    line("camera " + p, [a for a in blind if prof[a["game_id"]] == p])
print("SHOWN cards: share moved")
for name, key in (("method", lambda a: a["method"]), ("camera", lambda a: prof[a["game_id"]])):
    groups = defaultdict(list)
    for a in shown:
        if a.get("machine_x") is not None:
            groups[key(a)].append(a)
    for k, g in sorted(groups.items()):
        moved = [a for a in g if a["source"] == "coach"]
        print(f"  {name} {k:14s} n{len(g):3d} moved {len(moved) / len(g):4.0%}  moved by median {median(err(a) for a in moved) if moved else 0:.1f} ft")
ft = [a for a in done if a.get("x") is not None and math.hypot(a["x"] - WIDTH / 2, a["y"] - FT_LINE_Y) <= 2.5]
print("his dots within 2.5 ft of the middle of the line:", len(ft), Counter(a["method"] for a in ft), "by game", Counter(a["game_id"][11:30] for a in ft).most_common(10))
print("no-dot shots he placed", sum(a.get("machine_x") is None for a in done), "skipped of those", )

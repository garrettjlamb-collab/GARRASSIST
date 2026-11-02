"""The three low-camera games after his review: each game on its hidden cards, with the tolerance he asked for, beside the ten games reviewed before; and what he skipped."""
import json
import math
from collections import Counter, defaultdict
from statistics import median

from film.perceive.shooter_trust import good_enough
from film.review.dotreview import read_answers

THREE = {"2026-10-01-combs-vs-mdn-angle-1": "Combs", "2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1": "MDN vs CS", "2026-10-09-new-cactus-shadows-angle-1": "CS gym"}
rows = read_answers()
new = [a for a in rows if a["game_id"] in THREE]
print("answers on the three", len(new), "skipped", sum(bool(a.get("skipped")) for a in new), Counter(a.get("source") for a in new if not a.get("skipped")))
ts = sorted(a["at"] for a in new)
gaps = [b - a for a, b in zip(ts, ts[1:])]
print("median s", round(median(gaps), 1), "active min", round(sum(g for g in gaps if g < 120) / 60, 1))
print("skips by game", Counter(THREE[a["game_id"]] for a in new if a.get("skipped")))
print("skips", [(THREE[a["game_id"]], a["event_id"]) for a in new if a.get("skipped")][:40])


def line(name, g):
    g = [a for a in g if a.get("machine_x") is not None and not a.get("skipped")]
    if not g:
        print(f"  {name:30s} none")
        return
    e = [math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"]) for a in g]
    th = lambda x, y: math.hypot(x - 25, y - 5.25) > 19.75
    print(f"  {name:30s} n{len(g):3d} good {sum(good_enough((a['x'], a['y']), (a['machine_x'], a['machine_y'])) for a in g) / len(g):4.0%} median {median(e):4.1f} over10 {sum(v > 10 for v in e):3d} ({sum(v > 10 for v in e) / len(g):3.0%}) arc {sum(th(a['x'], a['y']) == th(a['machine_x'], a['machine_y']) for a in g) / len(g):4.0%}")


print("HIDDEN")
for k, name in THREE.items():
    line(name, [a for a in new if a["game_id"] == k and a.get("blind")])
    line("  trusted", [a for a in new if a["game_id"] == k and a.get("blind") and (a.get("trust") or {}).get("trusted")])
line("all three", [a for a in new if a.get("blind")])
line("  trusted", [a for a in new if a.get("blind") and (a.get("trust") or {}).get("trusted")])
line("  not trusted", [a for a in new if a.get("blind") and a.get("trust") and not a["trust"]["trusted"]])
line("ten earlier games, hidden", [a for a in rows if a["game_id"] not in THREE and a.get("blind")])
print("SHOWN (not independent)")
for k, name in THREE.items():
    g = [a for a in new if a["game_id"] == k and not a.get("blind") and not a.get("skipped") and a.get("machine_x") is not None]
    if g:
        print(f"  {name:12s} shown {len(g)} kept {sum(a['source'] == 'approved' for a in g)} moved {sum(a['source'] == 'coach' for a in g)}")
        line(name, g)
    print(f"  {name:12s} no machine dot, placed by him", sum(a.get("machine_x") is None and not a.get("skipped") for a in new if a["game_id"] == k))
ft = [a for a in new if not a.get("skipped") and math.hypot(a["x"] - 25, a["y"] - 19) <= 2.5]
print("his dots at the line", Counter(THREE[a["game_id"]] for a in ft))

"""Where the error lives, and a fair comparison. Good enough: a dot of his in the paint counts the machine's within 6 ft (he nudged layups and says the paint has room);
elsewhere within 4 ft and on the same side of the arc. Machine dots are the ones on disk now, for every shot. Same shots in every line."""
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from film.common.paths import game_dir
from film.interpret.events import load_events
from film.interpret.zones import is_three, zone_of
from shooter_rank import FEATURES, fit, label

data = pickle.loads(Path(sys.argv[1]).read_bytes())
events = {}
for g in {d["game"] for d in data}:
    events[g] = {e.id: e for e in load_events(game_dir(g))}
for d in data:
    e = events[d["game"]][d["id"]]
    d["now"] = None if e.x is None else (e.x, e.y)
    d["method"] = (e.signals.get("location") or {}).get("method")


def paint(x, y):
    return zone_of(x, y) in ("paint", "restricted")


def good(d, p):
    if p is None:
        return False
    gap = math.hypot(d["x"] - p[0], d["y"] - p[1])
    if paint(d["x"], d["y"]):
        return gap <= 6.0
    return gap <= 4.0 and is_three(d["x"], d["y"]) == is_three(*p)


def line(name, rows, key):
    got = [d for d in rows if d.get(key) is not None]
    if not got:
        print(f"  {name:30s} none")
        return
    e = [math.hypot(d["x"] - d[key][0], d["y"] - d[key][1]) for d in got]
    print(f"  {name:30s} n{len(got):3d} good {sum(good(d, d[key]) for d in got) / len(got):4.0%} median {median(e):4.1f} over10 {sum(v > 10 for v in e) / len(e):4.0%} arc {sum(is_three(d['x'], d['y']) == is_three(*d[key]) for d in got) / len(got):4.0%}")


his_paint = [d for d in data if paint(d["x"], d["y"])]
print("his dots in the paint", len(his_paint), "of", len(data), "| zones", Counter(zone_of(d["x"], d["y"]) for d in data).most_common(5))
withdot = [d for d in data if d["now"] is not None]
print("WHERE THE ERROR LIVES, shots with a dot on disk")
line("machine now", withdot, "now")
line("  his dot in the paint", [d for d in withdot if paint(d["x"], d["y"])], "now")
line("  his dot outside", [d for d in withdot if not paint(d["x"], d["y"])], "now")
line("  hidden only", [d for d in withdot if d["independent"]], "now")
for d in data:
    near = sorted(d["c"], key=lambda c: math.hypot(c["floor"][0] - d["x"], c["floor"][1] - d["y"]))
    d["oracle"] = near[0]["floor"]
line("best candidate, if chosen right", withdot, "oracle")
line("  hidden only", [d for d in withdot if d["independent"]], "oracle")

# Model with the machine's own pick and the pose ranking as features: can it beat the machine, not just copy it?
EXTRA = ["near_now", "top_arms_ball", "ab_margin"]
for d in data:
    ab = np.array([c["f"][FEATURES.index("arms_ball")] for c in d["c"]])
    top = ab.max() if len(ab) else 0.0
    for c, v in zip(d["c"], ab):
        near = 0.0 if d["now"] is None else math.exp(-math.hypot(c["floor"][0] - d["now"][0], c["floor"][1] - d["now"][1]) / 2.0)
        c["base"] = c.get("base", c["f"][:len(FEATURES)])
        c["f"] = list(c["base"]) + [near, float(v == top and top > 0), float(v - np.sort(ab)[-2]) if len(ab) > 1 and v == top else 0.0]
    d["label"] = label(d, 6.0 if paint(d["x"], d["y"]) else 4.0)
print("shots with a candidate within tolerance", sum(d["label"] is not None for d in data), "of", len(data))
import shooter_rank
shooter_rank.FEATURES = FEATURES + EXTRA
for g in sorted({d["game"] for d in data}):
    w = fit([d for d in data if d["game"] != g])
    for d in data:
        if d["game"] != g:
            continue
        p = np.array([c["f"] for c in d["c"]]) @ w
        probs = np.exp(p - p.max())
        probs /= probs.sum()
        i = int(np.argmax(p))
        d["model"], d["prob"] = d["c"][i]["floor"], float(probs[i])
print("weights", {k: round(float(v), 2) for k, v in zip(shooter_rank.FEATURES, fit(data))})
print("LEAVE ONE GAME OUT")
for name, rows in (("all with a dot", withdot), ("hidden with a dot", [d for d in withdot if d["independent"]])):
    print(name)
    line("machine", rows, "now")
    line("model", rows, "model")
print("no machine dot, model's guess")
line("model", [d for d in data if d["now"] is None], "model")
print("TRUST: does the model's certainty say which dots are good? shown as the machine's dot, on shots with one")
for lo, hi in ((0, .4), (.4, .6), (.6, .8), (.8, 1.01)):
    rows = [d for d in withdot if lo <= d["prob"] < hi]
    line(f"prob {lo}-{hi}", rows, "now")
    line("   hidden", [d for d in rows if d["independent"]], "now")
agree = [d for d in withdot if math.hypot(d["now"][0] - d["model"][0], d["now"][1] - d["model"][1]) <= 3.0]
differ = [d for d in withdot if d not in agree]
line("model and machine agree", agree, "now")
line("   hidden", [d for d in agree if d["independent"]], "now")
line("they differ: machine", differ, "now")
line("they differ: model", differ, "model")
for lo in (0.5, 0.6, 0.7):
    rows = [d for d in agree if d["prob"] >= lo]
    line(f"agree and prob>={lo}", rows, "now")
    line("   hidden", [d for d in rows if d["independent"]], "now")

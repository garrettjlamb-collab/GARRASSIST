"""Fresh test of the trust claim: answers given since the trust model was written, with the tolerance he asked for."""
import json
import math
from collections import Counter
from statistics import median

from film.interpret.court import FT_LINE_Y, WIDTH
from film.perceive.shooter_trust import good_enough
from film.review.dotreview import read_answers

rows = [a for a in read_answers() if a.get("trust") and not a.get("skipped")]
print("answers since trust", len(rows), Counter(a.get("source") for a in rows), "skipped since", sum(bool(a.get("skipped")) for a in read_answers()))
ts = sorted(a["at"] for a in rows)
gaps = [b - a for a, b in zip(ts, ts[1:])]
print("median s", round(median(gaps), 1), "active min", round(sum(g for g in gaps if g < 120) / 60, 1))


def line(name, g):
    g = [a for a in g if a["machine_x"] is not None]
    if not g:
        print(f"  {name}: none")
        return
    e = [math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"]) for a in g]
    print(f"  {name:34s} n{len(g):3d} good {sum(good_enough((a['x'], a['y']), (a['machine_x'], a['machine_y'])) for a in g) / len(g):4.0%} median {median(e):4.1f} within3 {sum(v <= 3 for v in e) / len(e):4.0%} over10 {sum(v > 10 for v in e) / len(e):4.0%}  ({sum(v > 10 for v in e)})")


trusted = [a for a in rows if a["trust"]["trusted"]]
rest = [a for a in rows if not a["trust"]["trusted"]]
line("trusted, hidden: the fresh test", [a for a in trusted if a.get("blind")])
line("trusted, shown", [a for a in trusted if not a.get("blind")])
line("not trusted, hidden", [a for a in rest if a.get("blind")])
line("not trusted, shown (anchored)", [a for a in rest if not a.get("blind")])
print("  not trusted, shown: kept", sum(a["source"] == "approved" for a in rest if not a.get("blind")), "moved", sum(a["source"] == "coach" for a in rest if not a.get("blind")))
ft = [a for a in rows if math.hypot(a["x"] - WIDTH / 2, a["y"] - FT_LINE_Y) <= 2.5]
print("his dots within 2.5 ft of the line's middle:", len(ft))
for a in sorted([a for a in trusted if a.get("blind") and math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"]) > 6], key=lambda a: -math.hypot(a["x"] - a["machine_x"], a["y"] - a["machine_y"])):
    print("   miss", a["game_id"][11:30], a["event_id"], "his", a["x"], a["y"], "machine", a["machine_x"], a["machine_y"], "prob", a["trust"]["prob"])

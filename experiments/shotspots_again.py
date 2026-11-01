"""Run the location rules again over the kept looks of every game that has them, and score them against the scoreboard."""
import glob
from collections import Counter
from pathlib import Path

from film.interpret.events import load_events
from film.interpret.zones import is_three
from film.perceive.shotspots import place
from film.review.decisions import load_decisions

tot = Counter()
for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    place(gd.name, log=lambda m: None)
    dec = load_decisions(gd)
    conf = [e for e in load_events(gd) if (dec.get(e.id) or {}).get("status") == "confirmed" and e.signals.get("phase") != "warmup"]
    fl = [e for e in conf if e.x is not None and e.signals["location"]["method"] == "flight"]
    ft = [e for e in conf if e.x is not None and e.signals["location"]["method"] == "free-throw"]
    made = [e for e in fl if e.made and e.signals.get("points") in (2, 3)]
    r = sum(is_three(e.x, e.y) == (e.signals["points"] == 3) for e in made)
    p3 = [e for e in made if e.signals["points"] == 3]
    r3 = sum(is_three(e.x, e.y) for e in p3)
    print(f"{gd.name.replace('2026-10-09-new-marcos-de-niza-high-school-vs-', '')[:22]:22s} shots {len(conf):3d} flight {len(fl):3d} ft {len(ft):2d} none {len(conf) - len(fl) - len(ft):2d} | side {r}/{len(made)} threes {r3}/{len(p3)}")
    tot.update(c=len(conf), fl=len(fl), ft=len(ft), made=len(made), r=r, p3=len(p3), r3=r3)
print(dict(tot), "twos", f"{tot['r'] - tot['r3']}/{tot['made'] - tot['p3']}", "threes", f"{tot['r3']}/{tot['p3']}", "all", f"{tot['r']}/{tot['made']}")

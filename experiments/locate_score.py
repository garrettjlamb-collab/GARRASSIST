"""What's on disk now against the scoreboard, without changing anything: side of the arc by rule, by game, and how each
kind of dot does, so confidence and the path's marks can be judged."""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from film.interpret.events import Event
from film.interpret.zones import is_three
from film.perceive.shotspots import confirmed_shots

HELD = ("2025-02-05-arcadia", "2025-02-05-jv-vs-marcos")
buckets = defaultdict(list)
for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    shots = confirmed_shots(gd)
    key = "HELD" if gd.name.startswith(HELD) else "tuned"
    methods = Counter((e.signals.get("location") or {}).get("method") for e in shots)
    made = [e for e in shots if e.x is not None and e.made and e.signals.get("points") in (2, 3) and e.signals["location"]["method"] in ("flight", "path", "pose")]
    r = lambda es: f"{sum(is_three(e.x, e.y) == (e.signals['points'] == 3) for e in es)}/{len(es)}"
    print(f"{gd.name.replace('2026-10-09-new-marcos-de-niza-high-school-vs-', '')[:24]:24s} {key:5s} shots {len(shots):3d} {dict(methods)} 2s {r([e for e in made if e.signals['points'] == 2])} 3s {r([e for e in made if e.signals['points'] == 3])}")
    for e in made:
        ok = is_three(e.x, e.y) == (e.signals["points"] == 3)
        p = e.signals["points"]
        buckets[f"{key} {p}s"].append(ok)
        buckets[f"{key} method {e.signals['location']['method']} {p}s"].append(ok)
        for fl in ("rule-differs", "path-disagrees"):
            if fl in e.flags:
                buckets[f"{key} flag {fl} {p}s"].append(ok)
        band = "high" if (e.location_conf or 0) >= 0.8 else "mid" if (e.location_conf or 0) >= 0.5 else "low"
        buckets[f"{key} conf {band}"].append(ok)
        buckets[f"{key} conf {band} {p}s"].append(ok)
    for e in shots:
        if e.x is not None:
            buckets["dots " + key].append(1)
        buckets["shots " + key].append(1)
for k in sorted(buckets):
    v = buckets[k]
    print(f"  {k}: {sum(v)}/{len(v)}" + (f" {sum(v) / len(v):.0%}" if not k.startswith(("dots", "shots")) else ""))

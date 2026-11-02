"""Why so many shots on the low-camera games have no dot, and whether the non-player filter is taking real players there."""
from collections import Counter
from pathlib import Path

import numpy as np

from film.perceive.shotspots import confirmed_shots, read_looks

for g in ("2026-10-01-combs-vs-mdn-angle-1", "2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1", "2026-10-09-new-cactus-shadows-angle-1", "2026-10-01-prescott-vs-arcadia"):
    gd = Path("games") / g
    rows = {r["id"]: r for r in read_looks(gd)}
    shots = confirmed_shots(gd)
    flags = Counter()
    ball, rims, sb = [], [], 0
    for e in shots:
        if e.x is None:
            for f in e.flags:
                if f in ("no-flight", "no-shooter", "no-player-found"):
                    flags[f] += 1
            r = rows.get(e.id)
            if r:
                ball.append(sum(bool(f["ball"]) for f in r["frames"]))
                rims.append(len(r["rims"]))
                sb += r["rim_id"] < 0
    off = [v for r in rows.values() for v in (r.get("officials") or {}).values()]
    print(g[11:40], "no dot", sum(e.x is None for e in shots), "of", len(shots), dict(flags), "| ball frames median", np.median(ball) if ball else None, "of 34; rim boxes median", np.median(rims) if rims else None, "no rim boxes", sum(v == 0 for v in rims), "scoreboard-only cards", sb, "| people checked", len(off), "flagged", sum(v >= 0.5 for v in off))

"""Score the 3D path on every confirmed shot. Made free throws start at a known spot, so they measure the fit in feet; +2 and +3
say which side of the arc; and the on-court person nearest the path's start says whether it lands on someone."""
from __future__ import annotations

import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from film.common.manifest import Manifest
from film.interpret.camera import from_homography
from film.interpret.court import FT_LINE_Y, WIDTH, fold
from film.interpret.flight3d import fit_path, rims_of
from film.interpret.location import Look, ball_sightings
from film.interpret.zones import is_three
from film.perceive.court_run import load_mappings, mapping_at
from film.perceive.shotspots import NO_RIM, _rims_in_anchor, confirmed_shots, read_looks

from bisect import bisect_left
from dataclasses import replace


def smooth_at(maps, ts, t):
    """The mapping at t, blended between its neighbours: the camera turns between fits."""
    i = bisect_left(ts, t)
    if i == 0 or i >= len(ts):
        return mapping_at(maps, t)
    t0, m0, t1, m1 = ts[i - 1], maps[i - 1][1], ts[i], maps[i][1]
    if t1 - t0 > 2.5 or m0.length != m1.length:
        return mapping_at(maps, t)
    u = (t - t0) / max(t1 - t0, 1e-6)
    H = (1 - u) * m0.H / np.linalg.norm(m0.H) + u * m1.H * np.sign(np.sum(m0.H * m1.H)) / np.linalg.norm(m1.H)
    return replace(m0, H=H)


SMOOTH = "--no-smooth" not in sys.argv
Z = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--z=")), 8.0))
RADIUS = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--r=")), 6.0))
HELD = ("2025-02-05-arcadia", "2025-02-05-jv-vs-marcos")
tot = defaultdict(list)
for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    ts = [t for t, _ in maps]
    at = (lambda t: smooth_at(maps, ts, t)) if SMOOTH else (lambda t: mapping_at(maps, t))
    rows = {r["id"]: r for r in read_looks(gd)}
    scale = m.width / 1920
    res = defaultdict(list)
    for e in confirmed_shots(gd):
        r = rows.get(e.id)
        if r is None:
            continue
        looks = [Look(x["t"], at(x["t"]), [tuple(b) for b in x["people"]], [tuple(b) for b in x["ball"]]) for x in r["frames"]]
        anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
        if anchor is None:
            continue
        cam = from_homography(anchor.H, m.width, m.height)
        if cam is None:
            continue
        rims = rims_of(anchor.length)
        s = ball_sightings(looks, scale)
        lo, hi = (e.t - 0.6, e.t + 0.4) if r["rim_id"] >= NO_RIM else (looks[0].t + 0.5, looks[-1].t)
        p = fit_path(cam, s, rims, lo, hi, anchor.length, scale, release_z=Z)
        pts = e.signals.get("points") if e.made else None
        old = (e.x, e.y) if e.x is not None and e.signals["location"]["method"] == "flight" else None
        if pts == 1:                         # the current rule on a free throw, without the rule that puts it on the line
            from film.interpret.location import locate
            o = locate(looks, _rims_in_anchor(r, looks), arrival_t=e.t if r["rim_id"] >= NO_RIM else None, scale=scale)
            if o.x is not None:
                ox, oy = fold(o.x, o.y, anchor.length)
                res["ftold"].append(float(np.hypot(ox - WIDTH / 2, oy - FT_LINE_Y)))
        res["all"].append(p is not None)
        res["old"].append(old is not None)
        new = None
        if p is not None:
            x, y = fold(*p.spot, anchor.length)
            if pts == 1:
                res["ft"].append(float(np.hypot(x - WIDTH / 2, y - FT_LINE_Y)))
            look = min(looks, key=lambda k: abs(k.t - p.release_t))
            best = None
            if look.mapping is not None:
                for b in look.people:
                    a = look.mapping.to_court((b[0] + b[2]) / 2, b[3])
                    if a and look.mapping.contains((b[0] + b[2]) / 2, b[3], margin=3.0):
                        d = float(np.hypot(a[0] - p.spot[0], a[1] - p.spot[1]))
                        if best is None or d < best[0]:
                            best = (d, a)
            if best is not None:
                res["person"].append(best[0])
                if best[0] <= RADIUS:
                    new = fold(*best[1], anchor.length)
                    res["snapped"].append(1)
            if pts == 1 and best is not None and best[0] <= RADIUS:
                res["ftperson"].append(float(np.hypot(new[0] - WIDTH / 2, new[1] - FT_LINE_Y)))
        hybrid = new if new is not None else old
        for name_, spot in (("o", old), ("h", hybrid), ("p", new)):
            if spot is not None and pts in (2, 3):
                res[name_ + str(pts)].append(is_three(*spot) == (pts == 3))
        res["any"].append(hybrid is not None)
        if old is not None and new is not None:
            gap = float(np.hypot(old[0] - new[0], old[1] - new[1]))
            res["gap"].append(gap)
            if pts in (2, 3):
                okold, oknew = is_three(*old) == (pts == 3), is_three(*new) == (pts == 3)
                band = "agree" if gap <= 5 else "differ"
                res[f"{band}_{pts}_old"].append(okold)
                res[f"{band}_{pts}_new"].append(oknew)
    name = gd.name.replace("2026-10-09-new-marcos-de-niza-high-school-vs-", "")[:24]
    held = gd.name.startswith(HELD)
    g = lambda k: f"{sum(res[k])}/{len(res[k])}"
    ft, fp, pe = res["ft"], res["ftperson"], res["person"]
    print(f"{name:24s} {'HELD' if held else '    '} fit {g('all')} | FT ball n{len(ft)} {np.median(ft) if ft else 0:.1f} ft, person n{len(fp)} {np.median(fp) if fp else 0:.1f} | old 2s {g('o2')} 3s {g('o3')} | path-only 2s {g('p2')} 3s {g('p3')} | hybrid 2s {g('h2')} 3s {g('h3')} | near {np.median(pe) if pe else 0:.1f}")
    key = "held" if held else "tuned"
    for k, v in res.items():
        tot[key + k] += v
for key in ("tuned", "held"):
    gg = lambda k: f"{sum(tot[key + k])}/{len(tot[key + k])}"
    pass
    g = lambda k: f"{sum(tot[key + k])}/{len(tot[key + k])}"
    bal = lambda a, b: (np.mean(tot[key + a]) + np.mean(tot[key + b])) / 2 if tot[key + a] and tot[key + b] else 0
    fo, fp = tot[key + "ftold"], tot[key + "ftperson"]
    print(f"{key:5s} z{Z:g} r{RADIUS:g} FT feet: rule n{len(fo)} {np.median(fo) if fo else 0:.1f} (within 3 ft {np.mean(np.array(fo) <= 3) if fo else 0:.0%}), path-picked n{len(fp)} {np.median(fp) if fp else 0:.1f} ({np.mean(np.array(fp) <= 3) if fp else 0:.0%}) | rule 2s {g('o2')} 3s {g('o3')} avg {bal('o2', 'o3'):.0%} | hybrid 2s {g('h2')} 3s {g('h3')} avg {bal('h2', 'h3'):.0%} | dots {g('any')}")

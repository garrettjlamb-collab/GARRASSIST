"""Does the camera recovered from a court mapping put the rim where the film shows it? A rim is 10 ft up over the basket, which
nothing in the court fit knows. For rim boxes seen near shots: pixels between the box's centre and the nearer projected rim."""
import glob
from pathlib import Path

import numpy as np

from film.common.manifest import Manifest
from film.interpret.camera import from_homography
from film.interpret.court import BASKET, WIDTH
from film.perceive.court_run import load_mappings, mapping_at
from film.perceive.shotspots import read_looks

for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    errs, focal, height, feet = [], [], [], []
    for r in read_looks(gd):
        for t, rid, x1, y1, x2, y2 in r["rims"][::5]:
            mp = mapping_at(maps, t)
            if mp is None:
                continue
            c = from_homography(mp.H, m.width, m.height)
            if c is None:
                continue
            L = mp.length
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best = None
            for rim in ((BASKET[0], BASKET[1], 10.0), (WIDTH - BASKET[0], L - BASKET[1], 10.0)):
                p = c.project(rim)
                if p is not None:
                    d = float(np.hypot(p[0] - cx, p[1] - cy))
                    if best is None or d < best[0]:
                        best = (d, p)
            if best is None:
                continue
            errs.append(best[0] * 1920 / m.width)
            focal.append(c.focal)
            height.append(c.centre[2])
    name = gd.name.replace("2026-10-09-new-marcos-de-niza-high-school-vs-", "")[:26]
    if errs:
        print(f"{name:26s} n {len(errs):5d} rim error px median {np.median(errs):6.1f} p90 {np.percentile(errs, 90):6.1f} | focal median {np.median(focal):7.0f} | camera height ft median {np.median(height):5.1f} p10-90 {np.percentile(height, 10):.0f}-{np.percentile(height, 90):.0f}")
    else:
        print(name, "no rims")

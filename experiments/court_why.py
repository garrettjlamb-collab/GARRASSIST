"""Why so few frames get a court on the low-camera games: of evenly spaced frames, how many line up with a reference, how many marks that carries, and which step refuses."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from film.common.cloud import game_film, local_source
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.court import fit_mapping
from film.perceive.court_refs import FIT_TOL_PX_AT_1080P, _fit, load_references, measure_length, place_points
from film.perceive.court_run import load_mappings
from film.perceive.register import features, masks_for, match
from film.train.prelabel import first_frame_at

for game in sys.argv[1:]:
    gd = game_dir(game)
    m = Manifest.load(gd)
    masks = masks_for(m)
    refs = load_references(game, masks)
    L = measure_length(refs) or 84.0
    print(game[11:45], "refs", [(r.image_id[-6:], len(r.points)) for r in refs], "length", L, "placed", len(load_mappings(gd)))
    c = Counter()
    inl, npts, errs = [], [], []
    with local_source(game_film(game), log=lambda s: None) as source:
        for t in np.linspace(m.duration_s * 0.1, m.duration_s * 0.9, 40):
            frame = first_frame_at(Path(source), float(t), m.fps)
            if frame is None:
                continue
            f = features(frame, masks)
            lined = [r for r in refs if match(r.features, f) is not None]
            pts, best = place_points(f, refs)
            inl.append(best)
            npts.append(len(pts))
            if not lined:
                c["no reference lines up"] += 1
                continue
            c[f"lines up with {len(lined)}"] += 1
            tol = max(3.0, FIT_TOL_PX_AT_1080P * m.width / 1920)
            strict = _fit(pts, m.width, m.height, L)
            loose = fit_mapping(pts, reproj_tol_px=30.0 * m.width / 1920, length=L)
            at8 = fit_mapping(pts, reproj_tol_px=tol, length=L)
            if at8 is not None:
                c["fits at 8 px"] += 1
                if not at8.contains(m.width / 2, m.height / 2, margin=15.0):
                    c["  refused only because the middle of the picture is off the court"] += 1
                    low = [at8.contains(m.width * a, m.height * 0.8, margin=15.0) for a in (0.25, 0.5, 0.75)]
                    c["    floor low in the picture is on the court"] += any(low)
            if strict is not None:
                c["fit ok"] += 1
            elif loose is not None:
                c["refused at 8 px, fits at 30 px"] += 1
                errs.append(round(loose.error_px, 1))
            else:
                c["no fit even at 30 px"] += 1
    print("  ", dict(c), "median inliers", np.median(inl), "median marks", np.median(npts), "loose errors", sorted(errs)[:12])

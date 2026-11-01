"""How well a set of marks fits a court of each length: the marks' average distance from the fit, in pixels."""
import json
import sys

from film.interpret.court import fit_mapping

pts = {k: tuple(v) for k, v in json.loads(sys.argv[1]).items()}
for L in (74.0, 84.0, 94.0):
    m = fit_mapping(pts, reproj_tol_px=25, length=L)
    print(int(L), None if m is None else (m.points_used, round(m.error_px, 1)))

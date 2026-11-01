"""Score the location rules on a cached sample (experiments/shotspots_collect.py) against the scoreboard: +1 is a free
throw at a known spot, +2 and +3 say which side of the arc."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from film.common.paths import game_dir
from film.interpret.court import BASKET, FT_LINE_Y, WIDTH, Mapping
from film.interpret.location import Look, free_throw_setup, locate
from film.interpret.zones import is_three
from film.perceive.court_run import load_mappings, mapping_at
from film.interpret.court import fold


def _rims_in_anchor(r: dict, looks: list) -> list[tuple[float, float]]:
    """Every rim seen in the window, as the centre of its box carried into the anchor frame's pixels."""
    anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
    if anchor is None:
        return []
    inv, out = np.linalg.inv(anchor.H), []
    by = {k.t: k for k in looks}
    for t, rid, x1, y1, x2, y2 in r.get("rims", []):
        if r["rim_id"] >= 0 and rid != r["rim_id"]:
            continue
        k = min(looks, key=lambda q: abs(q.t - t))
        if k.mapping is None:
            continue
        v = inv @ (k.mapping.H @ np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0]))
        out.append((float(v[0] / v[2]), float(v[1] / v[2])))
    return out


def score(game_id: str, path: Path, verbose: bool = True) -> list[dict]:
    maps = load_mappings(game_dir(game_id))
    out = []
    for r in json.loads(path.read_text()):
        looks = [Look(f["t"], mapping_at(maps, f["t"]), [tuple(b) for b in f["people"]], [tuple(b) for b in f["ball"]]) for f in r["frames"]]
        rims = _rims_in_anchor(r, looks)
        s = locate(looks, rims, arrival_t=r["t"] if r["rim_id"] >= 0 else None)
        row = {"id": r["id"], "points": r["points"], "made": r["made"], "method": s.method, "rel": None if s.release_t is None else round(s.release_t - r["t"], 1),
               "pts": s.arc_points, "span": s.arc_span_s, "reach": s.reach_px, "x": None, "y": None, "flags": s.flags}
        if s.x is not None:
            length = maps[0][1].length
            hx, hy = fold(s.x, s.y, length)
            row.update(x=round(hx, 1), y=round(hy, 1), three=is_three(hx, hy), ft_err=round(float(np.hypot(hx - WIDTH / 2, hy - FT_LINE_Y)), 1))
        row["ft_setup"] = free_throw_setup(looks, r["t"])
        out.append(row)
        if verbose:
            print(row)
    return out


def summary(rows: list[dict]) -> None:
    done = [r for r in rows if r["x"] is not None]
    print(f"located {len(done)} of {len(rows)}")
    done = [r for r in done if r["made"]]
    rows = [r for r in rows if r["made"]]
    print("no flight", sum(r["flags"] == ["no-flight"] for r in rows), "far", sum("shooter-far-from-ball" in r["flags"] for r in rows))
    print("free-throw setup seen on: +1", sum(r["ft_setup"] for r in rows if r["points"] == 1), "of", sum(r["points"] == 1 for r in rows),
          "; +2/+3", sum(r["ft_setup"] for r in rows if r["points"] in (2, 3)), "of", sum(r["points"] in (2, 3) for r in rows))
    print([r["id"] for r in rows if r["points"] == 1 and not r["ft_setup"]], [r["id"] for r in rows if r["points"] in (2, 3) and r["ft_setup"]])
    wrong = [(r["id"], r["points"], r["x"], r["y"]) for r in done if r["points"] in (2, 3) and (r["points"] == 3) != r["three"]]
    print("wrong side", wrong)
    fts = [r["ft_err"] for r in done if r["points"] == 1]
    if fts:
        print(f"free throws {len(fts)} of {sum(r['points'] == 1 for r in rows)}: median {np.median(fts):.1f} ft from the line, within 3 ft {np.mean(np.array(fts) <= 3):.0%}")
    made = [r for r in done if r["points"] in (2, 3)]
    if made:
        ok = [(r["points"] == 3) == r["three"] for r in made]
        print(f"twos and threes {len(made)} of {sum(r['points'] in (2, 3) for r in rows)}: arc side right {np.mean(ok):.0%}")


if __name__ == "__main__":
    rows = score(sys.argv[1], Path(sys.argv[2]), verbose=len(sys.argv) < 4)
    summary(rows)

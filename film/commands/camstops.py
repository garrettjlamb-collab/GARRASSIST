from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("camstops", help="a film with no scoreboard from a camera that stopped at breaks: "
                                             "find each restart and line the live stat book up with each stretch")
    parser.add_argument("game_id")
    parser.add_argument("statbook", type=Path, help="the stat book's JSON for this game (logged live)")
    parser.add_argument("--off", action="store_true", help="remove camstops.json: back to matching shot cards")
    parser.set_defaults(func=run)


def _scan(source: Path, log=print) -> tuple[list, list, float]:
    """One pass over the film: frame jumps at ~10/s (for restarts) and the camera's left-right shift at ~5/s."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    jump_every = max(1, round(fps / 10))
    pan_every = jump_every * 2
    win = cv2.createHanningWindow((240, 135), cv2.CV_32F)
    diffs, pan = [], []
    prev_small = prev_pan = None
    i = 0
    while cap.grab():
        if i % jump_every == 0:
            ok, f = cap.retrieve()
            if ok:
                grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(grey, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
                if prev_small is not None:
                    diffs.append((i / fps, float(np.abs(small - prev_small).mean())))
                prev_small = small
                if i % pan_every == 0:
                    g = cv2.resize(grey, (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32)
                    if prev_pan is not None:
                        (dx, _), _ = cv2.phaseCorrelate(prev_pan, g, win)
                        pan.append((i / fps, float(dx)))
                    prev_pan = g
        if i % (fps * 600) < 1:
            log(f"  scanned {i / fps / 60:.0f} min")
        i += 1
    return diffs, pan, fps / pan_every


def run(args) -> int:
    import json

    from film.common.cloud import local_source
    from film.common.manifest import Manifest
    from film.common.paths import game_dir
    from film.export.statbook import CAMSTOPS_FILE
    from film.interpret.camstops import camera_position, find_cuts, fit_offsets
    from film.interpret.statbook import load_statbook

    gd = game_dir(args.game_id)
    path = gd / CAMSTOPS_FILE
    if args.off:
        path.unlink(missing_ok=True)
        print(f"removed {path}")
        return 0
    m = Manifest.load(gd)
    book = load_statbook(args.statbook)
    timed = [p.ms for p in book.plays if p.ms is not None]
    t0 = min(timed)
    shots = [((p.ms - t0) / 1000.0, 1 if p.team == "us" else -1, p.period)
             for p in book.plays if p.is_shot and not p.is_ft and p.ms is not None]
    source = Path(m.source)
    with local_source(source):
        diffs, pan, rate = _scan(source)
    cuts = find_cuts(diffs)
    bounds = [0.0] + cuts + [float(m.duration_s)]
    cam_t, cam_pos = camera_position(pan, bounds, rate)
    stretches = fit_offsets(bounds, cam_t, cam_pos, shots)
    path.write_text(json.dumps({"statbook": str(args.statbook), "cuts": cuts,
                                "stretches": [s.__dict__ for s in stretches]}, indent=1))
    print(f"{len(cuts)} restarts, {len(stretches)} stretches")
    for s in stretches:
        a = s.start - s.offset
        print(f"  film {s.start:7.1f}-{s.end:7.1f}  = logged {a / 60:5.1f}-{(a + s.end - s.start) / 60:5.1f} min  "
              f"offset {s.offset:7.1f}  shots {s.shots:3d}  fit {s.score:+.2f}")
    print(f"wrote {path}; now: film statbook {args.game_id} <statbook> --all-reels <folder>")
    return 0

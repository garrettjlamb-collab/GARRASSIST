from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("calibrate", help="draw the scoreboard cells on a frame and print what each decodes to")
    parser.add_argument("frame", type=Path, help="a JPEG frame at the profile's native resolution")
    parser.add_argument("--profile", default="hudl-tactical")
    parser.add_argument("--out", type=Path, default=Path("calibrate.png"))
    parser.set_defaults(func=run)


def run(args) -> int:
    import cv2

    from film.perceive.sevenseg import SEGMENT_ZONES, box_fraction, decode_cell, led_mask, lit_segments

    profile = json.loads((files("film.perceive.profiles") / f"{args.profile.replace('-', '_')}.json").read_text())
    frame = cv2.imread(str(args.frame))
    if frame is None:
        print(f"could not read {args.frame}")
        return 1
    canvas = frame.copy()
    for name, field in profile["fields"].items():
        mask = led_mask(frame, field["color"])
        for cell in field["cells"]:
            x, y, w, h = cell
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
            for zone in SEGMENT_ZONES.values():
                zx1, zy1, zx2, zy2 = zone
                cv2.rectangle(canvas, (int(x + zx1 * w), int(y + zy1 * h)), (int(x + zx2 * w), int(y + zy2 * h)), (255, 255, 0), 1)
            print(f"{name:11s} cell {cell}: lit={lit_segments(mask, cell) or '-':8s} digit={decode_cell(mask, cell)}")
        for key in ("sep_top", "sep_bottom"):
            if key in field:
                x1, y1, x2, y2 = field[key]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
                print(f"{name:11s} {key}: fraction={box_fraction(mask, field[key]):.2f}")
    x1, y1, x2, y2 = 0, 860, 360, 1080
    zoom = cv2.resize(canvas[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(args.out), zoom)
    print(f"wrote {args.out}")
    return 0

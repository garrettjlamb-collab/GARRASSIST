from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("glyphs", help="teach a profile its scoreboard font from clock values read by eye")
    parser.add_argument("game_id")
    parser.add_argument("labels", nargs="+", metavar="T=M:SS",
                        help="a film second and the clock it shows, e.g. 256=7:58 (cover every digit), or @file of them")
    parser.add_argument("--profile", help="the profile to write the glyphs for (default: the game's)")
    parser.set_defaults(func=run)


def run(args) -> int:
    import cv2

    from film.common.cloud import local_source
    from film.common.manifest import Manifest
    from film.common.paths import game_dir
    from film.perceive.glyphs import clock_digits, glyphs_path, harvest
    from film.perceive.scoreboard import load_profile

    m = Manifest.load(game_dir(args.game_id))
    name = args.profile or m.profile
    profile = load_profile(name)
    if not profile.get("glyphs"):
        print(f"profile {name} names no glyphs file; add \"glyphs\" and \"glyph_fields\" to it first")
        return 1
    items = []
    for item in args.labels:                                    # "@file" reads one label a line, # for comments
        if item.startswith("@"):
            lines = __import__("pathlib").Path(item[1:]).read_text().splitlines()
            items += [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        else:
            items.append(item)
    labels = []
    for item in items:
        t, value = item.split("=")
        clock_digits(value)                                     # refuse a malformed value before reading film
        labels.append((float(t), value))
    frames = []
    with local_source(__import__("pathlib").Path(m.source)) as film:
        cap = cv2.VideoCapture(str(film))
        for t, value in labels:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                print(f"no frame at {t}s")
                return 1
            frames.append((frame, value))
        cap.release()
    glyphs = harvest(frames, profile)
    out = glyphs_path(profile["glyphs"])
    glyphs.save(__import__("pathlib").Path(str(out)))
    seen = {k: sorted({d for d, _ in v}) for k, v in glyphs.examples.items()}
    print(f"wrote {out}: " + "; ".join(f"{k} digits {v}" for k, v in sorted(seen.items())))
    return 0

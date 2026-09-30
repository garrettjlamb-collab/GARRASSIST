from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("eval", help="score the machine against the coach's decisions and an optional list of true shot times")
    parser.add_argument("game_id")
    parser.add_argument("--reference", type=Path, default=None, help="text file, one shot time per line (seconds or M:SS of video time)")
    parser.add_argument("--tol", type=float, default=2.0, help="seconds of slack when matching times")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export.evaluate import evaluate, load_reference

    reference = load_reference(args.reference) if args.reference else None
    result = evaluate(args.game_id, reference_times=reference, tol_s=args.tol)
    width = max(len(k) for k in result)
    for key, value in result.items():
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"{key:{width}s}  {shown}")
    return 0

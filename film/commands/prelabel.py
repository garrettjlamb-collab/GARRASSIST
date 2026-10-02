from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("prelabel", help="queue frames, rim crops or rim montages for labeling, with proposed labels")
    parser.add_argument("what", choices=["rim", "ball", "rimclf"])
    parser.add_argument("game_id")
    parser.add_argument("--n", type=int, default=None, help="how many to queue (rim 150, ball 300, rimclf none-samples 40)")
    parser.add_argument("--after-shots", action="store_true",
                        help="ball only: crops from just after confirmed shots where the detector lost the ball")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    if args.what == "rim":
        from film.train.prelabel import prelabel_rims

        prelabel_rims(args.game_id, n=args.n or 150, device=args.device)
    elif args.what == "ball":
        from film.train.prelabel import prelabel_balls

        prelabel_balls(args.game_id, n=args.n or 300, device=args.device, after_shots=args.after_shots)
    else:
        from film.train.rimclf_dataset import build_rimclf_candidates

        build_rimclf_candidates(args.game_id, n_none=args.n or 40, device=args.device)
    print("next: uv run film review, then open the Label tab")
    return 0

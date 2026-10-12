from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("train", help="fine-tune a model from the labeled datasets and register a new version")
    parser.add_argument("what", choices=["rim", "ball", "rimclf", "picclf", "court"])
    parser.add_argument("--hold", default=None, help="picclf only: the game to score on (default: the one labeled last)")
    parser.add_argument("--epochs", type=int, default=None, help="detectors only, default 50")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.common.device import pick_device

    device = pick_device(args.device)
    if args.what == "picclf":
        from film.train.train_yolo import train_picclf_version

        train_picclf_version(device, hold=args.hold, **({"epochs": args.epochs} if args.epochs else {}))
    elif args.what == "rimclf":
        from film.train.train_yolo import train_rimclf_version

        train_rimclf_version()
    else:
        from film.train.train_yolo import train_detector

        train_detector(args.what, device, epochs=args.epochs or 50)
    return 0

from __future__ import annotations

import argparse
import importlib

from film import __version__

COMMANDS = [
    "film.commands.doctor",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="film", description="Basketball film pipeline")
    parser.add_argument("--version", action="version", version=f"film {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    for module_name in COMMANDS:
        importlib.import_module(module_name).register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

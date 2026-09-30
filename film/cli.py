from __future__ import annotations

import argparse
import importlib
import os

import certifi

from film import __version__

COMMANDS = [
    "film.commands.doctor",
    "film.commands.ingest",
    "film.commands.calibrate",
    "film.commands.scoreboard",
    "film.commands.perceive",
    "film.commands.interpret",
    "film.commands.prelabel",
    "film.commands.train",
    "film.commands.clips",
    "film.commands.review",
    "film.commands.export",
    "film.commands.eval",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="film", description="Basketball film pipeline")
    parser.add_argument("--version", action="version", version=f"film {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    for module_name in COMMANDS:
        importlib.import_module(module_name).register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
import re
from pathlib import Path


def _root(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).resolve()


def games_dir() -> Path:
    return _root("FILM_GAMES_DIR", "games")


def models_dir() -> Path:
    return _root("FILM_MODELS_DIR", "models")


def datasets_dir() -> Path:
    return _root("FILM_DATASETS_DIR", "datasets")


def labels_dir() -> Path:
    return _root("FILM_LABELS_DIR", "labels")


def game_dir(game_id: str) -> Path:
    return games_dir() / game_id


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "game"


def game_id_for(source: Path, date: str) -> str:
    return f"{date}-{slugify(source.stem)}"

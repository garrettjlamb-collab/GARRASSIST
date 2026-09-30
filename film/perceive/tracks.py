from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["frame", "t", "kind", "track_id", "ref_id", "x1", "y1", "x2", "y2", "conf",
           "on_court", "court_x", "court_y", "team", "model_version"]
FLOAT_COLS = ["t", "x1", "y1", "x2", "y2", "conf", "court_x", "court_y"]
INT_COLS = ["frame", "track_id", "ref_id"]
TEXT_COLS = ["kind", "team", "model_version"]


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["on_court"] = df["on_court"].astype("boolean")
    for col in TEXT_COLS:
        df[col] = df[col].astype("string")
    return df


def tracks_path(game_dir: Path) -> Path:
    return Path(game_dir) / "tracks.parquet"


def read_tracks(game_dir: Path, kind: str | None = None) -> pd.DataFrame:
    path = tracks_path(game_dir)
    df = pd.read_parquet(path) if path.exists() else rows_to_frame([])
    if kind is not None:
        df = df[df["kind"] == kind]
    return df.reset_index(drop=True)


def replace_kind(game_dir: Path, df_new: pd.DataFrame, kind: str) -> Path:
    """Write tracks.parquet with this kind's rows replaced and every other kind kept."""
    existing = read_tracks(game_dir)
    existing = existing[existing["kind"] != kind]
    incoming = rows_to_frame(df_new.to_dict("records"))
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.sort_values(["frame", "kind"], kind="stable").reset_index(drop=True)
    Path(game_dir).mkdir(parents=True, exist_ok=True)
    merged.to_parquet(tracks_path(game_dir), index=False)
    return tracks_path(game_dir)

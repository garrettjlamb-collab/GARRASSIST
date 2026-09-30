import pandas as pd

from film.perceive.tracks import COLUMNS, read_tracks, replace_kind, rows_to_frame


def row(frame, kind, track_id=None, ref_id=None):
    return {"frame": frame, "t": frame / 10, "kind": kind, "track_id": track_id, "ref_id": ref_id,
            "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "conf": 0.5, "on_court": None,
            "court_x": None, "court_y": None, "team": None, "model_version": "fake@test"}


def test_rows_to_frame_has_every_column_and_types():
    df = rows_to_frame([row(0, "rim", track_id=1), {"frame": 1, "t": 0.1, "kind": "ball", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "conf": 0.2}])
    assert list(df.columns) == COLUMNS
    assert df["track_id"].dtype.name == "Int64" and pd.isna(df.loc[1, "track_id"])
    assert df["t"].dtype.name == "float64"
    assert rows_to_frame([]).empty and list(rows_to_frame([]).columns) == COLUMNS


def test_replace_kind_keeps_other_kinds(tmp_path):
    replace_kind(tmp_path, rows_to_frame([row(0, "rim", 1), row(1, "rim", 1)]), "rim")
    replace_kind(tmp_path, rows_to_frame([row(0, "ball", ref_id=1)]), "ball")
    assert len(read_tracks(tmp_path)) == 3
    replace_kind(tmp_path, rows_to_frame([row(5, "rim", 2)]), "rim")
    df = read_tracks(tmp_path)
    assert len(df) == 2 and set(df["kind"]) == {"rim", "ball"}
    assert read_tracks(tmp_path, "rim")["frame"].tolist() == [5]
    assert read_tracks(tmp_path / "missing").empty

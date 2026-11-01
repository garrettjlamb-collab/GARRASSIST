import json

import cv2
import numpy as np
import pytest

import film.train.datasets as ds
from film.common.paths import labels_dir
from film.interpret.court import _mirror_name, template
from film.perceive import court_draft
from synth_court import crop, frame, panorama
import synth_court

W, H = 1000, 500
BACK = np.array([[0.0, 9.0, 60.0], [8.0, 0.0, 40.0], [0.0, 0.0, 1.0]])   # feet -> pixels: along the court runs right
NEAR = ["near-corner-left", "near-corner-right", "near-lane-left", "near-lane-right", "near-ft-left", "near-ft-right"]


def _px(name):
    v = BACK @ np.array([*template()[name], 1.0])
    return [float(v[0] / v[2]), float(v[1] / v[2])]


def _handles(names=NEAR):
    return [{"name": n, "px": _px(n), "pinned": False} for n in names]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def _queue(image_id="g1_000100", image=None, **extra):
    ds.write_pending("court", image_id, np.zeros((H, W, 3), np.uint8) if image is None else image, [],
                     extra={"t": 10.0, "points": {}, "ref": True, **extra})
    return image_id


def test_a_draft_is_saved_and_a_picture_of_it_written():
    i = _queue()
    got = court_draft.set_draft(i, {n: _px(n) for n in NEAR}, length=84)
    assert got["kept"] == sorted(NEAR) and got["dropped"] == [] and got["error_px"] < 1
    assert cv2.imread(str(got["picture"])) is not None
    data = ds.read_pending("court", i)
    assert sorted(data["points"]) == sorted(NEAR) and data["length"] == 84 and data["draft"] == {"by": "claude"}


def test_a_bad_mark_is_reported_and_left_out_of_the_draft():
    i = _queue()
    points = {n: _px(n) for n in NEAR}
    points["near-lane-left"][1] += 120
    got = court_draft.set_draft(i, points)
    assert got["dropped"] == ["near-lane-left"]
    assert "near-lane-left" not in ds.read_pending("court", i)["points"]


def test_a_mirrored_draft_is_refused_and_not_saved():
    i = _queue()
    with pytest.raises(ValueError):
        court_draft.set_draft(i, {_mirror_name(n): _px(n) for n in NEAR})
    assert ds.read_pending("court", i)["points"] == {}


def test_draw_gives_the_lines_and_the_fit():
    got = court_draft.draw(_handles(), 94)
    assert got["ok"] is True and len(got["lines"]) > 10 and len(got["H"]) == 3


def test_draw_says_so_when_the_court_would_be_a_mirror():
    got = court_draft.draw([{"name": _mirror_name(n), "px": _px(n), "pinned": False} for n in NEAR], 94)
    assert got["ok"] is False and "lines" not in got


def test_a_pulled_spot_anywhere_on_the_floor_moves_the_court():
    a = court_draft.fit_of(_handles(), 94)
    spot = {"ft": [10.0, 30.0], "px": [400.0, 150.0], "pinned": True}
    b = court_draft.fit_of(_handles() + [spot], 94)
    x, y = b.to_court(400.0, 150.0)
    assert abs(x - 10) < 0.3 and abs(y - 30) < 0.3
    assert a.to_court(400.0, 150.0) != b.to_court(400.0, 150.0)


def test_approving_saves_every_point_in_view_through_the_fit():
    i = _queue(draft={"by": "claude"}, length=84)
    got = court_draft.approve(i, _handles(), 84)
    assert i not in ds.list_pending("court")
    _, saved, _, _ = ds.read_points("court", i)
    assert got["points"] == len(saved) and "near-arc-left" in saved and "mid-left" in saved    # more than the marks
    v = BACK @ np.array([*template(84)["mid-left"], 1.0])
    assert abs(saved["mid-left"][0] - v[0] / v[2]) < 1 and abs(saved["mid-left"][1] - v[1] / v[2]) < 1


def test_approving_logs_who_drew_it_and_how_far_it_moved():
    i = _queue(draft={"by": "claude"}, points={n: _px(n) for n in NEAR})
    moved = _handles()
    moved[0] = {"name": moved[0]["name"], "px": [moved[0]["px"][0] + 30, moved[0]["px"][1] + 40], "pinned": True}
    court_draft.approve(i, moved, 94, seconds=12.5)
    row = json.loads((labels_dir() / "court" / "review_log.jsonl").read_text().splitlines()[-1])
    assert row["image_id"] == i and row["by"] == "claude" and row["pulled"] == 1 and row["seconds"] == 12.5
    assert abs(row["moved_px"] - 50) < 1


def test_a_mirrored_approval_is_refused_and_stays_queued():
    i = _queue()
    with pytest.raises(ValueError):
        court_draft.approve(i, [{"name": _mirror_name(n), "px": _px(n), "pinned": False} for n in NEAR], 94)
    assert i in ds.list_pending("court")


def test_pending_pictures_say_who_drew_each():
    _queue("g1_000100")
    _queue("g1_000200", draft={"by": "claude"}, points={n: _px(n) for n in NEAR})
    got = court_draft.pending_pictures()
    assert [(p["image_id"], p["by"], p["points"]) for p in got] == [("g1_000100", None, 0), ("g1_000200", "claude", 6)]


def test_view_writes_a_gridded_picture_of_the_region():
    i = _queue()
    path = court_draft.view(i, region=(100, 100, 300, 200), scale=2.0)
    assert cv2.imread(str(path)).shape[:2] == (200, 400)


# ---- carrying: the same gym through a turning camera (tests/synth_court.py) ----

P_SMALL = np.array([[0.0, 14.0, 150.0], [6.0, 0.0, 100.0], [0.0, 0.0, 1.0]])
A, A2, FAR = crop(100.0, 60.0), crop(130.0, 60.0), crop(1000.0, 60.0)


@pytest.fixture
def pano(monkeypatch):
    monkeypatch.setattr(synth_court, "P", P_SMALL)
    return panorama()


def _pano_marks(H_crop, names):
    out = {}
    for n in names:
        x, y = template()[n]
        v = H_crop @ P_SMALL @ np.array([x, y, 1.0])
        out[n] = (float(v[0] / v[2]), float(v[1] / v[2]))
    return out


def test_an_approved_picture_redraws_a_queued_one_that_lines_up(pano, monkeypatch):
    monkeypatch.setattr(court_draft, "_masks", lambda game_id: [])
    ds.write_pending("court", "g1_000100", frame(pano, A), [], extra={"t": 10.0, "points": {}, "ref": True})
    ds.write_pending("court", "g1_000200", frame(pano, A2), [], extra={"t": 20.0, "points": {}, "ref": True})
    ds.write_pending("court", "g1_000300", frame(pano, FAR), [], extra={"t": 30.0, "points": {}, "ref": True})
    ds.accept_points("court", "g1_000100", _pano_marks(A, NEAR))
    assert court_draft.carry_drafts("g1") == 1
    near, far = ds.read_pending("court", "g1_000200"), ds.read_pending("court", "g1_000300")
    assert near["draft"]["by"] == "carried" and len(near["points"]) >= 4
    truth = _pano_marks(A2, ["near-ft-left"])["near-ft-left"]
    assert np.hypot(near["points"]["near-ft-left"][0] - truth[0], near["points"]["near-ft-left"][1] - truth[1]) < 3
    assert far["points"] == {} and "draft" not in far


def test_a_carried_mark_the_fit_disagrees_with_stays_out_of_the_draft(pano, monkeypatch):
    monkeypatch.setattr(court_draft, "_masks", lambda game_id: [])
    good = _pano_marks(A2, NEAR)
    carried = {**{n: tuple(p) for n, p in good.items()}, "mid-left": (-4000.0, 900.0)}
    monkeypatch.setattr(court_draft, "place_points", lambda f, refs: (carried, 100))
    ds.write_pending("court", "g1_000100", frame(pano, A), [], extra={"t": 10.0, "points": {}, "ref": True})
    ds.write_pending("court", "g1_000200", frame(pano, A2), [], extra={"t": 20.0, "points": {}, "ref": True})
    ds.accept_points("court", "g1_000100", _pano_marks(A, NEAR))
    assert court_draft.carry_drafts("g1") == 1
    data = ds.read_pending("court", "g1_000200")
    assert sorted(data["points"]) == sorted(NEAR)
    assert court_draft.approve("g1_000200", [{"name": n, "px": p, "pinned": False} for n, p in data["points"].items()], data["length"])["points"] > 0


def test_a_thin_carried_fit_does_not_replace_a_claude_draft(pano, monkeypatch):
    monkeypatch.setattr(court_draft, "_masks", lambda game_id: [])
    four = NEAR[:4]
    mine = {n: [1.0, 2.0] for n in NEAR}
    ds.write_pending("court", "g1_000100", frame(pano, A), [], extra={"t": 10.0, "points": {}, "ref": True})
    ds.write_pending("court", "g1_000200", frame(pano, A2), [], extra={"t": 20.0, "points": mine, "ref": True, "draft": {"by": "claude"}})
    ds.accept_points("court", "g1_000100", _pano_marks(A, four))
    assert court_draft.carry_drafts("g1") == 0
    assert ds.read_pending("court", "g1_000200")["points"] == mine


def test_the_command_saves_a_draft_and_says_what_it_left_out(capsys):
    import argparse

    from film.commands import court_draft as command

    i = _queue()
    parser = argparse.ArgumentParser()
    command.register(parser.add_subparsers())
    points = {n: _px(n) for n in NEAR}
    points["near-lane-left"][1] += 120
    args = parser.parse_args(["court-draft", "set", i, "--points", json.dumps(points), "--length", "84"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "saved 5 marks at 84 ft" in out and "left out, look again: near-lane-left" in out
    args = parser.parse_args(["court-draft", "list"])
    assert args.func(args) == 0
    assert f"{i}  5 marks by claude, 84 ft" in capsys.readouterr().out

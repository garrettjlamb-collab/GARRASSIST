from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.perceive.tracks import replace_kind, rows_to_frame
from film.train import datasets as ds
from film.review.decisions import save_decision
from film.train.rimclf_dataset import build_rimclf_candidates, rim_box_at


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def _rims(gd, frames=range(30)):
    rows = [{"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "ref_id": None, "x1": 100.0, "y1": 60.0, "x2": 150.0,
             "y2": 90.0, "conf": 0.9, "on_court": None, "court_x": None, "court_y": None, "team": None, "model_version": "x"}
            for f in frames]
    replace_kind(gd, rows_to_frame(rows), "rim")


def _event(eid, t, kind="shot", points=0, scoreboard=None):
    signals = {"points": points, "score_delta": [points, 0] if points else None,
               "scoreboard": scoreboard or ("made" if points else "unknown")}
    return Event(id=eid, type=kind, t=t, rim_id=1, quarter=None, clock_s=None, team=None, made=None, confidence=0.5,
                 clip={"start": t - 1, "end": t + 1}, signals=signals, flags=[])


def test_rim_box_at_uses_nearest_frame(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd, frames=[0, 1, 2, 20])
    from film.perceive.tracks import read_tracks

    rims = read_tracks(gd, "rim")
    assert rim_box_at(rims, 2) == (100.0, 60.0, 150.0, 90.0)
    assert rim_box_at(rims, 6) == (100.0, 60.0, 150.0, 90.0)     # frame 2 is within max_gap
    assert rim_box_at(rims, 40) is None                          # nearest is frame 20, too far
    assert rim_box_at(rims, 20, rim_id=1) == (100.0, 60.0, 150.0, 90.0) and rim_box_at(rims, 20, rim_id=9) is None


def test_build_candidates_routes_made_pending_and_none(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd)
    save_events(gd, [_event("shot-000010-r1", 1.0, points=2), _event("shot-000020-r1", 2.0), _event("shot-000025-r1", 2.5, kind="unknown")])
    counts = build_rimclf_candidates(m.game_id, n_none=2, seed=1, margin_s=0.2, exclusion_s=0.1, log=lambda *a: None)
    assert counts == {"made_auto": 1, "pending": 2, "none_auto": 2, "already": 0, "no_rim": 0, "from_review": 0}
    root = tmp_path / "datasets" / "rimclf"
    assert (root / "val" / "made" / "shot-000010-r1.jpg").exists()
    assert sorted(p.parent.name for p in root.glob("*/none/*.jpg")) == ["none", "none"]
    pending = {i: ds.read_pending("rimclf", i) for i in ds.list_pending("rimclf")}
    assert pending["shot-000020-r1"]["suggested"] == "missed"
    assert pending["shot-000025-r1"]["suggested"] == "none"


def test_ambiguous_or_one_point_verdict_is_not_auto_made(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd)
    save_events(gd, [_event("shot-000010-r1", 1.0, points=2, scoreboard="ambiguous"), _event("shot-000020-r1", 2.0, points=1)])
    counts = build_rimclf_candidates(m.game_id, n_none=0, margin_s=0.2, log=lambda *a: None)
    assert counts == {"made_auto": 0, "pending": 2, "none_auto": 0, "already": 0, "no_rim": 0, "from_review": 0}


def test_rerun_adds_nothing_and_counts_no_rim(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd, frames=range(15))                                   # no rim box near the 2.5 s event
    save_events(gd, [_event("shot-000010-r1", 1.0, points=2), _event("shot-000020-r1", 2.0), _event("shot-000025-r1", 2.5, kind="unknown")])
    first = build_rimclf_candidates(m.game_id, n_none=1, seed=1, margin_s=0.2, exclusion_s=0.1, log=lambda *a: None)
    assert first["no_rim"] == 1 and first["made_auto"] == 1 and first["pending"] == 1 and first["none_auto"] == 1
    root = tmp_path / "datasets" / "rimclf"
    before = sorted(str(p) for p in root.rglob("*.jpg")) + ds.list_pending("rimclf")
    second = build_rimclf_candidates(m.game_id, n_none=1, seed=1, margin_s=0.2, exclusion_s=0.1, log=lambda *a: None)
    assert second["made_auto"] == 0 and second["pending"] == 0 and second["none_auto"] == 0 and second["already"] == 3
    assert sorted(str(p) for p in root.rglob("*.jpg")) + ds.list_pending("rimclf") == before


def test_review_decisions_route_montages_before_the_scoreboard_rule(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd)
    save_events(gd, [
        _event("shot-000005-r1", 0.5),                                    # confirmed, coach says made
        _event("shot-000008-r1", 0.8, points=2),                          # confirmed, coach says missed despite the scoreboard
        _event("shot-000011-r1", 1.1, points=2),                          # confirmed, made still unknown -> pending, suggest made
        _event("shot-000014-r1", 1.4),                                    # confirmed, made unknown, no verdict -> pending, suggest missed
        _event("shot-000017-r1", 1.7, points=2),                          # rejected -> none, not auto made
        _event("shot-000020-r1", 2.0, points=3),                          # unsure -> pending
        _event("shot-000023-r1", 2.3, points=2),                          # no decision -> the old rule: auto made
    ])
    save_decision(gd, "shot-000005-r1", {"status": "confirmed", "made": True})
    save_decision(gd, "shot-000008-r1", {"status": "confirmed", "made": False})
    save_decision(gd, "shot-000011-r1", {"status": "confirmed"})
    save_decision(gd, "shot-000014-r1", {"status": "confirmed", "made": None})
    save_decision(gd, "shot-000017-r1", {"status": "rejected"})
    save_decision(gd, "shot-000020-r1", {"status": "unsure"})
    counts = build_rimclf_candidates(m.game_id, n_none=0, margin_s=0.2, log=lambda *a: None)
    assert counts == {"made_auto": 1, "pending": 3, "none_auto": 0, "already": 0, "no_rim": 0, "from_review": 3}
    placed = {p.stem: p.parent.name for p in (tmp_path / "datasets" / "rimclf").glob("*/*/*.jpg")}
    assert placed == {"shot-000005-r1": "made", "shot-000008-r1": "missed", "shot-000017-r1": "none", "shot-000023-r1": "made"}
    pending = {i: ds.read_pending("rimclf", i) for i in ds.list_pending("rimclf")}
    assert {i: p["suggested"] for i, p in pending.items()} == {
        "shot-000011-r1": "made", "shot-000014-r1": "missed", "shot-000020-r1": "missed"}


def test_scoreboard_only_events_have_no_rim_and_are_skipped(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd)
    sb = _event("shot-000010-sb", 1.0, points=2)
    sb.rim_id = -1
    save_events(gd, [sb])
    save_decision(gd, "shot-000010-sb", {"status": "confirmed", "made": True})
    counts = build_rimclf_candidates(m.game_id, n_none=0, margin_s=0.2, log=lambda *a: None)
    assert counts["no_rim"] == 1 and counts["from_review"] == 0 and counts["made_auto"] == 0
    assert not list((tmp_path / "datasets").rglob("*.jpg")) and ds.list_pending("rimclf") == []


def test_a_pending_montage_is_promoted_once_the_coach_decides(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd)
    save_events(gd, [_event("shot-000010-r1", 1.0), _event("shot-000015-r1", 1.5), _event("shot-000020-r1", 2.0)])
    first = build_rimclf_candidates(m.game_id, n_none=0, margin_s=0.2, log=lambda *a: None)
    assert first["pending"] == 3
    save_decision(gd, "shot-000010-r1", {"status": "confirmed", "made": True})
    save_decision(gd, "shot-000015-r1", {"status": "rejected"})
    save_decision(gd, "shot-000020-r1", {"status": "unsure"})
    second = build_rimclf_candidates(m.game_id, n_none=0, margin_s=0.2, log=lambda *a: None)
    assert second["from_review"] == 2 and second["already"] == 1 and second["pending"] == 0
    placed = {p.stem: p.parent.name for p in (tmp_path / "datasets" / "rimclf").glob("*/*/*.jpg")}
    assert placed == {"shot-000010-r1": "made", "shot-000015-r1": "none"}
    assert ds.list_pending("rimclf") == ["shot-000020-r1"]


def test_a_rejected_warmup_card_waits_for_a_label_instead_of_teaching_none(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd)
    warm_made, warm_rejected = _event("shot-000005-r1", 0.5), _event("shot-000008-r1", 0.8)
    warm_made.flags, warm_rejected.flags = ["warmup"], ["warmup"]
    save_events(gd, [warm_made, warm_rejected])
    save_decision(gd, "shot-000005-r1", {"status": "confirmed", "made": True})   # a make is a make, warmup or not
    save_decision(gd, "shot-000008-r1", {"status": "rejected"})                  # may only mean "not the game"
    build_rimclf_candidates(m.game_id, n_none=0, margin_s=0.2, log=lambda *a: None)
    placed = {p.stem: p.parent.name for p in (tmp_path / "datasets" / "rimclf").glob("*/*/*.jpg")}
    assert placed == {"shot-000005-r1": "made"}
    assert ds.list_pending("rimclf") == ["shot-000008-r1"]

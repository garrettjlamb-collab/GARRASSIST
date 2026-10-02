from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.perceive.detectors import Box, FakeDetector
from film.train import datasets as ds
from film.train.prelabel import prelabel_balls, prelabel_rims, sample_times


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def test_sample_times():
    assert sample_times(100, 3) == [30.0, 50.0, 70.0]
    assert sample_times(50, 3) == []


def test_prelabel_rims_writes_pending(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    det = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim"), Box(0, 0, 9, 9, 0.9, "person")])
    n = prelabel_rims(m.game_id, n=4, detector=det, margin_s=0.5, log=lambda *a: None)
    assert n == 4
    ids = ds.list_pending("rim")
    assert len(ids) == 4 and all(i.startswith(m.game_id) for i in ids)
    assert [b["cls"] for b in ds.read_pending("rim", ids[0])["boxes"]] == ["rim"]


def test_prelabel_balls_samples_around_arrivals(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    ev = Event(id="shot-000010-r1", type="shot", t=1.0, rim_id=1, quarter=None, clock_s=None, team=None, made=None,
               confidence=0.3, clip={"start": 0.0, "end": 2.5}, signals={}, flags=[])
    save_events(game_dir(m.game_id), [ev])
    rims = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim")])
    balls = FakeDetector(fn=lambda img, i: [Box(5, 5, 15, 15, 0.6, "ball")])
    n = prelabel_balls(m.game_id, n=10, rim_detector=rims, ball_detector=balls, margin_s=0.5, log=lambda *a: None)
    assert n == 5                                              # five offsets around the one arrival
    data = ds.read_pending("ball", ds.list_pending("ball")[0])
    assert data["width"] == 240 and data["height"] == 240        # the rim crop, not the full frame
    assert data["boxes"][0]["cls"] == "ball"


def _shot(eid, t, rim_id=1, source="motion"):
    return Event(id=eid, type="shot", t=t, rim_id=rim_id, quarter=None, clock_s=None, team=None, made=None,
                 confidence=0.5, clip={"start": 0.0, "end": 2.5}, signals={"source": source}, flags=[])


def test_spread_picks_evenly_across_the_whole_list():
    from film.train.prelabel import spread
    assert spread(list(range(10)), 20) == list(range(10))
    assert spread(list(range(100)), 5) == [0, 25, 50, 74, 99]


def test_prelabel_balls_uses_only_shots_the_coach_confirmed(synthetic_video, tmp_path, monkeypatch):
    from film.review.decisions import save_decision
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    save_events(gd, [_shot("shot-000005-r1", 0.5), _shot("shot-000020-r1", 2.0)])
    save_decision(gd, "shot-000005-r1", {"status": "rejected"})
    save_decision(gd, "shot-000020-r1", {"status": "confirmed", "made": True})
    rims = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim")])
    balls = FakeDetector(fn=lambda img, i: [])
    n = prelabel_balls(m.game_id, n=300, rim_detector=rims, ball_detector=balls, log=lambda *a: None)
    times = sorted(ds.read_pending("ball", i)["t"] for i in ds.list_pending("ball"))
    assert n == len(times) and n >= 6                    # a dense run of frames around the one confirmed shot
    assert min(times) >= 0.6 and max(times) <= 3.0       # 2.0 - 1.4 ... 2.0 + 1.0, nothing around the rejected card
    assert all(abs((b - a) - 0.2) < 1e-6 for a, b in zip(times, times[1:]))


def test_prelabel_balls_does_not_requeue_crops_already_labeled(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    save_events(game_dir(m.game_id), [_shot("shot-000010-r1", 1.0)])
    rims = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim")])
    balls = FakeDetector(fn=lambda img, i: [])
    first = prelabel_balls(m.game_id, n=10, rim_detector=rims, ball_detector=balls, log=lambda *a: None)
    ids = ds.list_pending("ball")
    ds.accept_boxes("ball", ids[0], [Box(5, 5, 15, 15, 1.0, "ball")])
    again = prelabel_balls(m.game_id, n=10, rim_detector=rims, ball_detector=balls, log=lambda *a: None)
    assert first == 5 and again == 0
    assert len(ds.list_pending("ball")) == 4
    skipped = ds.list_pending("ball")[0]
    ds.remove_pending("ball", skipped)
    ds.mark_skipped("ball", skipped)
    assert prelabel_balls(m.game_id, n=10, rim_detector=rims, ball_detector=balls, log=lambda *a: None) == 0   # a skip is remembered


def _after_shot_game(synthetic_video, lost_at=(0.2, 0.4, 0.6, 0.8), made=True):
    """A reviewed game with one confirmed shot at 1.0 s whose stored ball path loses the ball at the given moments."""
    import json
    from film.perceive.tracks import replace_kind, rows_to_frame
    from film.review.decisions import save_decision

    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    rows = [{"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "x1": 100.0, "y1": 60.0, "x2": 150.0, "y2": 90.0, "conf": 0.9}
            for f in range(30)]
    replace_kind(gd, rows_to_frame(rows), "rim")
    save_events(gd, [_shot("shot-000010-r1", 1.0, source="ball")])
    save_decision(gd, "shot-000010-r1", {"status": "confirmed", "made": made})
    frames = [{"dt": round(-0.2 + 0.2 * i, 2), "balls": [] if round(-0.2 + 0.2 * i, 2) in lost_at else [[0.0, 0.3, 0.8]]}
              for i in range(9)]                                # -0.2 ... 1.4
    (gd / "ballpaths.json").write_text(json.dumps({"detector": "ball@x", "paths": {"10-r1": {"rim_w": 50.0, "frames": frames}}}))
    return m, gd


def test_after_shot_samples_are_the_moments_the_detector_lost_the_ball(synthetic_video, tmp_path, monkeypatch):
    from film.train.prelabel import after_shot_samples

    _env(monkeypatch, tmp_path)
    m, gd = _after_shot_game(synthetic_video, lost_at=(-0.2, 0.4, 0.8, 1.2))
    samples = after_shot_samples(gd, m.fps)
    assert samples == [(1.4, 1), (1.8, 1)]           # after the arrival and within a second of it; not -0.2, not 1.2


def test_after_shot_samples_take_more_from_a_make_than_from_a_miss(synthetic_video, tmp_path, monkeypatch):
    from film.train.prelabel import after_shot_samples

    _env(monkeypatch, tmp_path)
    lost = (0.2, 0.4, 0.6, 0.8, 1.0)
    m, gd = _after_shot_game(synthetic_video, lost_at=lost, made=True)
    assert len(after_shot_samples(gd, m.fps)) == 4
    from film.review.decisions import save_decision
    save_decision(gd, "shot-000010-r1", {"status": "confirmed", "made": False})
    assert len(after_shot_samples(gd, m.fps)) == 2


def test_prelabel_balls_after_shots_queues_the_crop_the_path_reader_sees(synthetic_video, tmp_path, monkeypatch):
    from film.interpret.ballpath import path_crop_box

    _env(monkeypatch, tmp_path)
    m, gd = _after_shot_game(synthetic_video, lost_at=(0.4, 0.8))
    balls = FakeDetector(fn=lambda img, i: [Box(5, 5, 15, 15, 0.3, "ball")])
    n = prelabel_balls(m.game_id, n=10, ball_detector=balls, after_shots=True, log=lambda *a: None)
    assert balls.calls == 0                                    # its weak guesses here are mostly wrong: propose nothing
    ids = ds.list_pending("ball")
    assert n == 2 and ids == [f"{m.game_id}_0000140_p1", f"{m.game_id}_0000180_p1"]
    x1, y1, x2, y2 = path_crop_box((100.0, 60.0, 150.0, 90.0), 320, 240)
    pending = ds.read_pending("ball", ids[0])
    assert (pending["width"], pending["height"]) == (x2 - x1, y2 - y1) and pending["boxes"] == []
    assert prelabel_balls(m.game_id, n=10, ball_detector=balls, after_shots=True, log=lambda *a: None) == 0   # not queued twice


def test_prelabel_rims_proposes_nothing_when_no_rim_model_fits_the_camera(synthetic_video, tmp_path, monkeypatch):
    """The open-vocabulary fallback finds no rims on this film (it boxes the ball, or nothing): queue bare frames."""
    from film.perceive.detectors import YoloDetector

    _env(monkeypatch, tmp_path)
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))

    def boom(*a, **k):
        raise AssertionError("the open-vocabulary model must not be loaded to propose rims")

    monkeypatch.setattr(YoloDetector, "open_vocab", classmethod(boom))
    m = ingest(synthetic_video, date="2026-01-01", profile="nfhs-auto")
    assert prelabel_rims(m.game_id, n=3, margin_s=0.5, log=lambda *a: None) == 3
    assert all(ds.read_pending("rim", i)["boxes"] == [] for i in ds.list_pending("rim"))

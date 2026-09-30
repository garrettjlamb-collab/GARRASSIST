import cv2

from film.common.paths import game_dir
from film.common.video import probe
from film.export.clips import build_reel, clip_path, clips_for_game, cut_clip
from film.ingest.run import ingest
from film.interpret.events import Event, save_events


def test_cut_clip_duration(synthetic_video, tmp_path):
    out = cut_clip(synthetic_video, 0.5, 1.5, tmp_path / "c" / "clip.mp4")
    assert out.exists()
    assert 0.8 <= probe(out).duration_s <= 1.3


def test_clips_for_game_and_reel(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i * 0.5, rim_id=1, quarter=1, clock_s=100.0, team=None,
                    made=bool(i % 2), confidence=0.5, clip={"start": 0.5 + i * 0.5, "end": 1.5 + i * 0.5}, signals={}, flags=[])
              for i in range(2)]
    save_events(gd, events)
    assert clips_for_game(m.game_id, log=lambda *a: None) == 2
    assert clip_path(gd, "shot-000000-r1").exists()
    assert clips_for_game(m.game_id, log=lambda *a: None) == 0          # already cut
    reel = build_reel([(clip_path(gd, e.id), f"Q{e.quarter} {'MADE' if e.made else 'MISS'}") for e in events], tmp_path / "reel.mp4")
    cap = cv2.VideoCapture(str(reel))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 10

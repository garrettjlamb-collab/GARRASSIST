from pathlib import Path

import cv2

from film.ingest.profile import detect_profile, overlay_correlation, overlay_features

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"
HUDL = sorted(p.name for p in FIXTURES.glob("hudl_tactical_*.jpg"))
NFHS_AUTO = sorted(p.name for p in FIXTURES.glob("nfhs_auto_*.jpg"))


def load(name: str):
    img = cv2.imread(str(FIXTURES / name))
    assert img is not None, name
    return img


def test_fixture_sets_are_complete():
    assert len(HUDL) == 5 and len(NFHS_AUTO) == 3


def test_hudl_frames_show_the_overlay():
    for name in HUDL:
        led, panel = overlay_features(load(name))
        assert led > 0.02, (name, led)
        assert panel > 0.35, (name, panel)
        assert overlay_correlation(load(name)) > 0.4, name
    assert overlay_correlation(load("hudl_tactical_t2400.jpg")) > 0.95


def test_nfhs_auto_frames_do_not():
    for name in NFHS_AUTO:
        assert overlay_correlation(load(name)) < 0.2, name
    assert overlay_features(load("nfhs_auto_t1500.jpg"))[1] < 0.05


def test_broadcast_region_out_of_frame():
    frame = load("nfhs_broadcast_t1500.jpg")  # 1280x720
    assert overlay_features(frame) == (0.0, 0.0)
    assert overlay_correlation(frame) == 0.0


def test_detect_profile():
    assert detect_profile([load(n) for n in HUDL]) == "hudl-tactical"
    assert detect_profile([load(n) for n in NFHS_AUTO]) == "unknown"
    assert detect_profile([]) == "unknown"

from pathlib import Path

import cv2

from film.ingest.profile import detect_profile, overlay_features

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"


def load(name: str):
    img = cv2.imread(str(FIXTURES / name))
    assert img is not None, name
    return img


def test_hudl_frames_show_the_overlay():
    for name in ("hudl_tactical_t1500.jpg", "hudl_tactical_t2400.jpg", "hudl_tactical_t4800.jpg"):
        led, panel = overlay_features(load(name))
        assert led > 0.02, (name, led)
        assert panel > 0.35, (name, panel)


def test_other_frames_do_not():
    led, panel = overlay_features(load("nfhs_auto_t1500.jpg"))
    assert not (led > 0.02 and panel > 0.35), (led, panel)
    assert overlay_features(load("nfhs_broadcast_t1500.jpg")) == (0.0, 0.0)  # 1280x720, region out of frame


def test_detect_profile():
    hudl = [load("hudl_tactical_t1500.jpg"), load("hudl_tactical_t2400.jpg"), load("hudl_tactical_t600.jpg")]
    assert detect_profile(hudl) == "hudl-tactical"
    assert detect_profile([load("nfhs_auto_t1500.jpg")]) == "unknown"
    assert detect_profile([]) == "unknown"

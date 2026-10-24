import cv2
import numpy as np

from film.common.paths import game_dir
from film.export.court_sheet import court_lines, court_sheet, draw_court
from film.ingest.run import ingest
from film.interpret.court import TEMPLATE, fit_mapping
from film.perceive.court_run import mapping_row, write_mappings


def _pix(name):
    x, y = TEMPLATE[name]
    return (x * 2 + 40, 200 - y * 2)                     # 2 px a foot, the near end at the bottom of a 320 x 240 frame


def _mapping():
    return fit_mapping({n: _pix(n) for n in TEMPLATE})


def test_court_lines_cover_both_ends_and_the_outline():
    lines = court_lines()
    allpts = np.vstack(lines)
    assert allpts[:, 0].min() == 0.0 and allpts[:, 0].max() == 50.0
    assert allpts[:, 1].min() == 0.0 and allpts[:, 1].max() == 94.0
    assert len(lines) == 6 + 2 * 5


def test_lines_are_drawn_where_the_mapping_puts_the_court():
    img = draw_court(np.zeros((240, 320, 3), np.uint8), _mapping(), thickness=1)
    for name in ("near-corner-left", "near-ft-left", "near-lane-right"):
        x, y = (int(round(v)) for v in _pix(name))
        patch = img[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        assert (patch[..., 2] > 80).any(), name                       # anti-aliased: some red+green there
    assert not img[120:125, 5:10].any()                                # off the court, nothing


def test_court_sheet_tiles_frames_from_court_parquet(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="handheld", fps=10.0)
    gd = game_dir(m.game_id)
    write_mappings(gd, [mapping_row(t, 10.0, _mapping(), "ref") for t in (0.0, 1.0, 2.0)])
    path = court_sheet(m.game_id, n=12, cols=3, tile_width=320, log=lambda *a: None)
    sheet = cv2.imread(str(path))
    assert sheet.shape[:2] == (240, 960)

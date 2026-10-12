# tests/test_court_run.py
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.court import TEMPLATE
from film.perceive.court_run import court_pass, load_mappings, mapping_at
from film.perceive.courtfinder import FakeCourtFinder


def _pix(names):
    return {n: (TEMPLATE[n][0] * 10 + 100, 500 - TEMPLATE[n][1] * 10) for n in names}


def test_court_pass_writes_mappings_once_a_second_and_the_manifest(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="hudl-mdn", fps=10.0)           # 3 s of film
    gd = game_dir(m.game_id)
    good = _pix(["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"])
    finder = FakeCourtFinder([good, {}, good])
    info = court_pass(m.game_id, finder=finder, log=lambda *a: None)
    assert info["status"] == "done" and info["step_s"] == 1.0 and info["frames"] == 3 and info["fitted"] == 2
    maps = load_mappings(gd)
    assert [round(t) for t, _ in maps] == [0, 2]
    assert mapping_at(maps, 0.4) is not None and mapping_at(maps, 1.0) is not None       # the gap is bridged
    assert mapping_at(maps, 1.0).end == "near"
    assert mapping_at(maps, 5.0) is None                                                 # three seconds past the last fit
    assert Manifest.load(gd).model_versions["court"] == "court@fake"


def test_handheld_film_is_fitted_twice_a_second(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="handheld", fps=10.0)
    finder = FakeCourtFinder([_pix(["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right"])])
    info = court_pass(m.game_id, finder=finder, log=lambda *a: None)
    assert info["step_s"] == 0.5 and info["frames"] == 6 and info["fitted"] == 6

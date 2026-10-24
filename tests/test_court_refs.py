import numpy as np
import pytest

import film.train.datasets as ds
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.perceive.court_refs import check_references, load_references, reference_ids, refs_pass
from film.perceive.court_run import load_mappings
from synth_court import court_points_in, crop, feet_of, frame, panorama
import synth_court

# A court small enough in the panorama that one frame sees most of an end, the way a sideline camera does:
# y (along) at 14 px/ft from x = 150, x (across) at 6 px/ft from y = 100. Frames start at y = 60 and see it all.
P_SMALL = np.array([[0.0, 14.0, 150.0], [6.0, 0.0, 100.0], [0.0, 0.0, 1.0]])
A, B = crop(100.0, 60.0), crop(1000.0, 60.0)          # the near end (pixels 100-740) and the far end (1000-1640)


@pytest.fixture(autouse=True)
def small_court(monkeypatch):
    monkeypatch.setattr(synth_court, "P", P_SMALL)


@pytest.fixture(scope="module")
def pano():
    return panorama()


def _env(monkeypatch, tmp_path):
    for env, sub in (("FILM_GAMES_DIR", "games"), ("FILM_DATASETS_DIR", "datasets"), ("FILM_LABELS_DIR", "labels")):
        monkeypatch.setenv(env, str(tmp_path / sub))


def _game(synthetic_video, date="2026-01-01"):
    return ingest(synthetic_video, date=date, profile="handheld", fps=10.0).game_id


def _add_ref(game_id, t, img, points):
    image_id = f"{game_id}_{int(round(t * 10)):06d}"
    ds.write_pending("court", image_id, img, [], extra={"t": t, "points": {}})
    ds.accept_points("court", image_id, points)
    return image_id


def _close(mapping, G, pixels, tol_ft=0.3):
    for px, py in pixels:
        got, want = mapping.to_court(px, py), feet_of(G, px, py)
        assert got is not None and np.hypot(got[0] - want[0], got[1] - want[1]) < tol_ft, (px, py, got, want)


def test_reference_ids_are_marked_frames_of_that_game_only(synthetic_video, tmp_path, monkeypatch, pano):
    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    img = frame(pano, A)
    _add_ref(g, 10.0, img, court_points_in(A))
    ds.write_pending("court", f"{g}_000200a", img, [], extra={"t": 20.0, "points": {}})     # machine-made, not a reference
    ds.accept_points("court", f"{g}_000200a", court_points_in(A))
    ds.write_pending("court", f"{g}-other_000300", img, [], extra={"t": 30.0, "points": {}})
    ds.accept_points("court", f"{g}-other_000300", court_points_in(A))
    assert reference_ids(g) == [f"{g}_000100"]
    refs = load_references(g, [])
    assert len(refs) == 1 and refs[0].t == 10.0 and len(refs[0].points) == len(court_points_in(A))


def test_frames_are_placed_from_a_reference_then_carried(synthetic_video, tmp_path, monkeypatch, pano):
    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    _add_ref(g, 10.0, frame(pano, A), court_points_in(A))
    shots = [(0.0, crop(120.0, 70.0, zoom=1.1)), (1.0, crop(500.0, 60.0)), (2.0, crop(860.0, 60.0))]
    info = refs_pass(g, frames=[(t, frame(pano, G)) for t, G in shots], log=lambda *a: None)
    assert info["fitted"] == 3 and info["from_ref"] == 2 and info["chained"] == 1
    maps = load_mappings(game_dir(g))
    for (t, G), (tm, mapping) in zip(shots, maps):
        assert tm == t
        _close(mapping, G, [(320, 180), (100, 300), (540, 80)])
    assert maps[2][1].conf < maps[1][1].conf                                        # a carried court is trusted less
    assert Manifest.load(game_dir(g)).model_versions["court"] == f"refs@{g}:1"


def test_points_from_two_references_pin_one_fit(synthetic_video, tmp_path, monkeypatch, pano):
    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    _add_ref(g, 10.0, frame(pano, A), court_points_in(A))
    _add_ref(g, 20.0, frame(pano, B), court_points_in(B))
    wide = crop(300.0, 40.0, zoom=0.6)                                              # sees both refs' halves of the gym
    info = refs_pass(g, frames=[(0.0, frame(pano, wide))], log=lambda *a: None)
    assert info["from_ref"] == 1
    (_, mapping), = load_mappings(game_dir(g))
    assert mapping.points_used > len(court_points_in(A))
    _close(mapping, wide, [(320, 180), (60, 200), (600, 150)])


def test_carrying_stops_thirty_seconds_after_the_last_reference(synthetic_video, tmp_path, monkeypatch, pano):
    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    _add_ref(g, 10.0, frame(pano, A), court_points_in(A))
    shots = [(0.0, crop(500.0, 60.0)), (40.0, crop(860.0, 60.0))]
    info = refs_pass(g, frames=[(t, frame(pano, G)) for t, G in shots], log=lambda *a: None)
    assert info["fitted"] == 1 and info["chained"] == 0


def test_another_games_references_place_this_one(synthetic_video, tmp_path, monkeypatch, pano):
    _env(monkeypatch, tmp_path)
    g1, g2 = _game(synthetic_video), _game(synthetic_video, date="2026-01-02")
    _add_ref(g1, 10.0, frame(pano, A), court_points_in(A))
    info = refs_pass(g2, refs_from=g1, frames=[(0.0, frame(pano, crop(150.0, 60.0)))], log=lambda *a: None)
    assert info["fitted"] == 1
    with pytest.raises(RuntimeError):
        refs_pass(g1, refs_from=g2, frames=[], log=lambda *a: None)                 # g2 has no references


def test_reference_fits_become_training_labels_with_only_points_in_view(synthetic_video, tmp_path, monkeypatch, pano):
    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    _add_ref(g, 10.0, frame(pano, A), court_points_in(A))
    shots = [(0.0, crop(120.0, 70.0)), (1.5, crop(130.0, 62.0))]
    refs_pass(g, frames=[(t, frame(pano, G)) for t, G in shots], train_frames=2, log=lambda *a: None)
    made = sorted(i for i in (p.stem for p in (tmp_path / "datasets" / "court" / "images").rglob("*.jpg")) if i.endswith("a"))
    assert made == [f"{g}_000000a", f"{g}_000015a"]
    _, points, w, h = ds.read_points("court", made[1])
    assert len(points) >= 4 and all(0 <= x < w and 0 <= y < h for x, y in points.values())
    want = court_points_in(crop(130.0, 62.0))
    for name, (x, y) in points.items():
        assert np.hypot(x - want[name][0], y - want[name][1]) < 2.0
    assert reference_ids(g) == [f"{g}_000100"]                                       # still the one reference


def test_check_references_measures_how_well_they_agree(synthetic_video, tmp_path, monkeypatch, pano):
    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    A2 = crop(120.0, 65.0)
    _add_ref(g, 10.0, frame(pano, A), court_points_in(A))
    slipped = court_points_in(A2)
    slipped["near-ft-top"] = (slipped["near-ft-top"][0] + 20.0, slipped["near-ft-top"][1])
    _add_ref(g, 20.0, frame(pano, A2), slipped)
    report = check_references(g)
    assert [r["t"] for r in report] == [10.0, 20.0] and all(r["lined_up"] for r in report)
    assert all(r["shared"] >= 4 for r in report)
    assert all(r["median_px"] < 1.5 for r in report)
    assert all(r["worst"] == "near-ft-top" and 18.0 < r["max_px"] < 22.0 for r in report)


def test_a_few_references_cover_a_panning_game(synthetic_video, tmp_path, monkeypatch, pano):
    from film.perceive.court_refs import choose_references

    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    samples = [(10.0 * (i + 1), frame(pano, crop(300.0 * i, 60.0))) for i in range(6)]   # neighbours overlap by half
    times, share = choose_references(g, samples=samples, log=lambda *a: None)
    assert share == 1.0 and 1 <= len(times) <= 3


def test_a_marked_reference_counts_and_a_skipped_frame_is_never_asked_again(synthetic_video, tmp_path, monkeypatch, pano):
    from film.perceive.court_refs import choose_references

    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    samples = [(10.0 * (i + 1), frame(pano, crop(300.0 * i, 60.0))) for i in range(6)]
    _add_ref(g, 99.0, frame(pano, crop(300.0, 60.0)), court_points_in(crop(300.0, 60.0)))   # sees what sample 2 sees
    ds.mark_skipped("court", f"{g}_000500")                                                 # sample 5 (t = 50)
    times, share = choose_references(g, samples=samples, log=lambda *a: None)
    assert share == 1.0 and times == [40.0, 60.0]            # without the skipped middle frame it takes two


def test_the_court_length_is_measured_where_a_picture_holds_both_ends(synthetic_video, tmp_path, monkeypatch, pano):
    from film.perceive.court_refs import measure_length

    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    near, far, wide = crop(100.0, 60.0), crop(700.0, 60.0), crop(300.0, 40.0, zoom=0.6)
    for t, G in ((10.0, near), (20.0, far), (30.0, wide)):
        _add_ref(g, t, frame(pano, G), court_points_in(G, length=84.0))
    assert measure_length(load_references(g, [])) == 84.0
    shot = crop(650.0, 60.0)
    info = refs_pass(g, frames=[(0.0, frame(pano, shot))], log=lambda *a: None)
    assert info["length_ft"] == 84.0 and info["length_measured"] is True
    (_, mapping), = load_mappings(game_dir(g))
    assert mapping.length == 84.0
    _close(mapping, shot, [(320, 180), (100, 300), (540, 80)])


def test_one_end_alone_cannot_measure_the_length(synthetic_video, tmp_path, monkeypatch, pano):
    from film.perceive.court_refs import measure_length

    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    _add_ref(g, 10.0, frame(pano, A), court_points_in(A))
    assert measure_length(load_references(g, [])) is None


def test_one_wide_picture_that_covers_everything_still_gets_the_views_at_each_end(synthetic_video, tmp_path, monkeypatch, pano):
    from film.perceive.court_refs import choose_references

    _env(monkeypatch, tmp_path)
    g = _game(synthetic_video)
    wide = [(5.0, frame(pano, crop(0.0, 40.0, zoom=0.5)))]                        # sees pixels 0-1280 of the gym
    close = [(10.0 * (i + 1), frame(pano, crop(160.0 * i, 60.0))) for i in range(5)]  # each inside the wide view
    times, share = choose_references(g, samples=wide + close, log=lambda *a: None)
    assert share == 1.0 and len(times) >= 3 and {10.0, 50.0} <= set(times)       # both ends are asked for


def test_the_view_furthest_round_is_found_through_a_chain_of_neighbours(pano):
    from film.perceive.court_refs import _spread_picks
    from film.perceive.register import features, match

    feats = [(10.0 * i, features(frame(pano, crop(300.0 * i, 60.0)))) for i in range(6)]
    links = [set() for _ in feats]
    for i in range(6):
        for j in range(i + 1, 6):
            if match(feats[i][1], feats[j][1]) is not None:
                links[i].add(j)
                links[j].add(i)
    anchor = feats[0][1]
    assert match(feats[5][1], anchor) is None                       # the far view shares nothing with the anchor
    assert _spread_picks(feats, links, anchor, {1, 2, 3, 4, 5}) == [1, 5]

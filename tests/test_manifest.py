from film.common.manifest import Manifest


def _manifest():
    return Manifest(
        game_id="2025-02-05-test", source="/x/test.mp4", date="2025-02-05",
        duration_s=3.0, width=320, height=240, src_fps=30.0, fps=10.0, profile="unknown",
    )


def test_roundtrip(tmp_path):
    m = _manifest()
    m.mark("ingest", "done", samples=5)
    m.model_versions["rim"] = "rim@v1"
    path = m.save(tmp_path / "game")
    assert path.name == "manifest.json"
    loaded = Manifest.load(tmp_path / "game")
    assert loaded == m
    assert loaded.stages["ingest"]["status"] == "done"
    assert loaded.stages["ingest"]["samples"] == 5
    assert "at" in loaded.stages["ingest"]


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path, monkeypatch):
    import film.common.manifest as manifest_mod

    moves = []
    real_replace = manifest_mod.os.replace
    monkeypatch.setattr(manifest_mod.os, "replace", lambda a, b: moves.append((str(a), str(b))) or real_replace(a, b))
    m = _manifest()
    m.save(tmp_path / "game")
    assert moves == [(str(tmp_path / "game" / "manifest.json.tmp"), str(tmp_path / "game" / "manifest.json"))]
    m.mark("perceive", "done")
    m.save(tmp_path / "game")
    assert sorted(p.name for p in (tmp_path / "game").iterdir()) == ["manifest.json"]
    assert Manifest.load(tmp_path / "game").stages["perceive"]["status"] == "done"


def test_load_ignores_unknown_keys(tmp_path):
    import json
    from dataclasses import asdict

    m = _manifest()
    data = asdict(m) | {"written_by_a_newer_version": True}
    (tmp_path / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    assert Manifest.load(tmp_path) == m


def test_a_game_can_carry_a_title_for_people(tmp_path):
    from film.common.manifest import Manifest

    m = Manifest(game_id="g", source="x.mp4", date="2026-01-01", duration_s=1.0, width=1, height=1, src_fps=30.0, fps=10.0, profile="test")
    assert m.title == "g"                                          # the id until someone names it
    m.title = "MDN vs St. Mary's (Hudl)"
    m.save(tmp_path)
    assert Manifest.load(tmp_path).title == "MDN vs St. Mary's (Hudl)"

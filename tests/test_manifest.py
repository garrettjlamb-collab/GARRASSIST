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

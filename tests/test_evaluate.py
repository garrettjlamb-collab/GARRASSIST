from film.common.paths import game_dir
from film.export.evaluate import evaluate, load_reference, match_times
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.review.decisions import save_decision


def test_load_reference(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text("# shots\n90\n25:00\nt=1500.4\n\n")
    assert load_reference(path) == [90.0, 1500.0, 1500.4]


def test_match_times():
    assert match_times([10.0, 20.0, 30.0], [11.0, 29.0, 50.0]) == (2, 1, 1)
    assert match_times([], [1.0]) == (0, 0, 1)


def test_evaluate(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    made = [True, True, False, None]
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=10.0 * (i + 1), rim_id=1, quarter=1, clock_s=None, team=None,
                    made=made[i], confidence=0.5, clip={"start": 0, "end": 1}, signals={}, flags=[]) for i in range(4)]
    events.append(Event(id="unknown-1", type="unknown", t=99.0, rim_id=1, quarter=1, clock_s=None, team=None, made=None,
                        confidence=0.2, clip={"start": 0, "end": 1}, signals={}, flags=[]))
    save_events(gd, events)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed"})                  # machine made=True, coach agrees
    save_decision(gd, "shot-000001-r1", {"status": "confirmed", "made": False})   # coach flipped it
    save_decision(gd, "shot-000002-r1", {"status": "rejected"})
    save_decision(gd, "shot-000003-r1", {"status": "unsure"})
    result = evaluate(m.game_id)
    assert result["shots"] == 4 and result["confirmed"] == 2 and result["rejected"] == 1 and result["unsure"] == 1 and result["undecided"] == 0
    assert result["precision"] == 2 / 3
    assert result["made_compared"] == 2 and result["made_agreement"] == 0.5
    with_ref = evaluate(m.game_id, reference_times=[10.5, 20.0, 60.0])
    assert with_ref["reference"] == 3 and with_ref["matched"] == 2
    assert with_ref["recall"] == 2 / 3 and with_ref["precision_vs_reference"] == 2 / 3      # machine = confirmed + unsure = 10, 20, 40


def test_warmup_cards_do_not_count_against_precision(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=10.0 * (i + 1), rim_id=1, quarter=1, clock_s=None, team=None,
                    made=True, confidence=0.5, clip={"start": 0, "end": 1}, signals={},
                    flags=(["warmup"] if i else [])) for i in range(3)]
    save_events(gd, events)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed"})
    save_decision(gd, "shot-000001-r1", {"status": "rejected"})
    result = evaluate(m.game_id)
    assert result["shots"] == 1 and result["warmups"] == 2 and result["precision"] == 1.0 and result["undecided"] == 0

from film.interpret.phases import END_GRACE_S, game_phases, is_live, is_this_game
from film.perceive.scoreboard import ScoreboardRead


def countdown(t0: float, period: int, start_clock: float, seconds: int, home: int = 0, away: int = 0,
              step: float = 1.0) -> list[ScoreboardRead]:
    """One read per second with the clock running down from start_clock."""
    return [ScoreboardRead(t=t0 + i * step, period=period, clock_s=max(0.0, start_clock - i * step),
                           home=home, away=away, home_fouls=0, away_fouls=0, conf=1.0)
            for i in range(seconds)]


def blank(t0: float, seconds: int, period: int, home, away) -> list[ScoreboardRead]:
    return [ScoreboardRead(t=t0 + i, period=period, clock_s=None, home=home, away=away,
                           home_fouls=0, away_fouls=0, conf=0.75) for i in range(seconds)]


def jv_game() -> list[ScoreboardRead]:
    """Pregame countdown, two 480 s quarters with a 60 s break, a 600 s halftime, one more quarter,
    a post-game countdown, then the scoreboard reset to 0-0 for the next game."""
    tl = []
    tl += countdown(0, 1, 363, 360)                       # pregame warmups        0 - 359
    tl += blank(360, 5, 1, 0, 0)                         # dropout while the clock is set
    tl += countdown(365, 1, 480, 500, 0, 0)              # Q1                   365 - 864
    tl += countdown(865, 2, 60, 60, 20, 11)              # break, period already advanced
    tl += countdown(925, 2, 480, 500, 20, 11)            # Q2                   925 - 1424
    tl += countdown(1425, 2, 600, 600, 35, 23)           # halftime            1425 - 2024
    tl += countdown(2025, 3, 480, 500, 35, 23)           # Q3                  2025 - 2524
    tl += countdown(2525, 3, 650, 300, 53, 37)           # post-game countdown 2525 - 2824
    tl += blank(2825, 5, 1, 0, 0)                        # reset for the next game
    tl += countdown(2830, 1, 480, 100, 0, 0)             # next game's Q1 (not this game)
    return tl


def test_quarters_are_live_and_everything_else_is_not():
    phases = game_phases(jv_game())
    assert phases.quarter_length_s == 480
    for t in (400, 700, 950, 1300, 2100, 2500):
        assert is_live(phases, t), t
    for t in (10, 300, 362, 870, 920, 1500, 2000, 2600, 2900):
        assert not is_live(phases, t), t


def test_reads_with_no_clock_belong_to_the_running_segment():
    tl = countdown(0, 1, 480, 100) + blank(100, 20, 1, 4, 2) + countdown(120, 1, 360, 100, 4, 2)
    phases = game_phases(tl)
    assert is_live(phases, 110)                          # dropout inside a quarter stays live


def test_a_reset_after_a_dropout_still_starts_a_segment():
    tl = countdown(0, 1, 363, 300) + blank(300, 10, 1, 0, 0) + countdown(310, 1, 480, 300)
    tl += countdown(610, 1, 480, 300)                    # a second 480 start makes 480 the quarter length
    phases = game_phases(tl)
    assert not is_live(phases, 100) and is_live(phases, 400)


def test_empty_or_unreadable_timeline_treats_everything_as_live():
    assert is_live(game_phases([]), 123.0)
    assert is_live(game_phases(blank(0, 50, 1, None, None)), 10.0)


def test_a_buzzer_beater_just_after_the_reset_is_still_live():
    phases = game_phases(jv_game())
    assert is_live(phases, 865 + END_GRACE_S - 0.5)      # Q1's clock reset at 865
    assert not is_live(phases, 865 + END_GRACE_S + 0.5)


def test_this_game_ends_when_the_board_resets():
    phases = game_phases(jv_game())
    assert phases.game_over_t == 2825
    assert is_this_game(phases, 2600) and not is_this_game(phases, 2900)


def test_a_board_with_scores_and_periods_but_no_clock_gives_quarters_from_the_score_activity():
    """The Marcos de Niza skin: no clock read. Live play runs from just before the first score change to
    just after the last; the board's period says which quarter. Pregame (0-0 for ten minutes) and the busy
    gym after the final buzzer are not the game."""
    tl = blank(0, 600, 1, 0, 0)                                     # pregame, board shows P1 0-0
    tl += [ScoreboardRead(t=600 + i, period=1, clock_s=None, home=2 * (i // 60), away=i // 90, home_fouls=None, away_fouls=None, conf=0.75) for i in range(480)]
    tl += blank(1080, 60, 2, 16, 5)                                 # break
    tl += [ScoreboardRead(t=1140 + i, period=2, clock_s=None, home=16 + i // 50, away=5 + i // 40, home_fouls=None, away_fouls=None, conf=0.75) for i in range(480)]
    tl += blank(1620, 900, 2, 25, 17)                               # final score stays up for 15 minutes
    phases = game_phases(tl)
    assert phases.known and phases.quarter_length_s is None
    assert not is_live(phases, 300) and is_live(phases, 700) and is_live(phases, 1500)
    assert not is_live(phases, 1620 + 300)                          # five minutes after the last score change
    assert phases.quarters == {1: (phases.live[0][0], phases.live[0][1]), 2: (phases.live[1][0], phases.live[1][1])}

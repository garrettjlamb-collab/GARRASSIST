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


def test_a_buzzer_beater_just_after_the_buzzer_is_still_live():
    phases = game_phases(jv_game())
    assert is_live(phases, 845 + END_GRACE_S - 0.5)      # Q1's clock hit 0:00 at 845 and was held there
    assert not is_live(phases, 845 + END_GRACE_S + 0.5)  # until the break clock was set at 865


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
    tl += blank(1620, 600, 4, 25, 17)                               # halftime: no scoring; the period cell misreads as 4
    tl += [ScoreboardRead(t=2220 + i, period=3, clock_s=None, home=25 + i // 45, away=17 + i // 60, home_fouls=None, away_fouls=None, conf=0.75) for i in range(480)]
    tl += blank(2700, 900, 3, 35, 24)                               # final score stays up for 15 minutes
    phases = game_phases(tl)
    assert phases.known and phases.quarter_length_s is None
    assert not is_live(phases, 300) and is_live(phases, 700) and is_live(phases, 1500)
    assert not is_live(phases, 1620 + 300)                          # halftime: five minutes without a score change
    assert is_live(phases, 2400) and not is_live(phases, 2700 + 300)
    assert list(phases.quarters) == [1, 2, 3] and phases.quarters[2][0] < 1200 < phases.quarters[2][1]
    assert phases.live == list(phases.quarters.values())


def test_a_known_quarter_length_beats_the_pregame_countdown():
    """At Marcos de Niza the board counts down 9:00 over and over before the game, so the commonest long
    start is not the quarter length. The profile says the quarter is 8:00; a stretch is a quarter when it
    starts near that and some scoring happens in it. Reads come with gaps, so "near" is generous."""
    tl = []
    for k in range(4):
        tl += countdown(k * 560, 1, 540, 540)                      # four 9:00 pregame countdowns, no scoring
    tl += countdown(2240, 1, 472, 480, 0, 0)                        # Q1, the first read 8 s late (gap), scoring below
    for r in tl[-480:]:
        r.home = 2 * ((r.t - 2240) // 120)
    tl += countdown(2720, 2, 480, 480, 8, 3)                        # Q2 with scoring
    for r in tl[-480:]:
        r.away = 3 + (r.t - 2720) // 100
    tl += countdown(3200, 2, 480, 480, 8, 7)                        # a dead 8:00 countdown after the game: no scoring
    phases = game_phases(tl, quarter_length_s=480.0)
    assert phases.quarter_length_s == 480.0 and phases.known
    assert not is_live(phases, 1000) and is_live(phases, 2300) and is_live(phases, 2900) and not is_live(phases, 3400)
    assert len(phases.live) == 2


def test_an_isolated_misread_of_the_clock_does_not_start_a_new_stretch():
    """The Marcos de Niza minutes digit is half hidden: 8:00 reads as 9:00 now and then. One stray read
    an hour before or after its neighbours must not split a quarter into fragments."""
    tl = countdown(0, 1, 480, 480, 0, 0)
    for r in tl[1:]:
        r.home = 2 * (r.t // 120)
    tl[200].clock_s = tl[200].clock_s + 60.0                          # a lone 9:xx read mid-quarter
    tl[300].clock_s = tl[300].clock_s + 60.0
    phases = game_phases(tl, quarter_length_s=480.0)
    assert len(phases.live) == 1 and is_live(phases, 250) and is_live(phases, 400)


def _score(reads, every: int, start_t: float | None = None, home: bool = True):
    """Add a basket every `every` seconds to a run of reads (scoring is what makes a countdown a quarter)."""
    t0 = start_t if start_t is not None else reads[0].t
    for r in reads:
        pts = 2 * max(0, int((r.t - t0) // every))
        if home:
            r.home = (r.home or 0) + pts
        else:
            r.away = (r.away or 0) + pts
    return reads


def _static(t0: float, period: int, clock: float, seconds: int, home: int = 0, away: int = 0) -> list[ScoreboardRead]:
    return [
        ScoreboardRead(t=t0 + i, period=period, clock_s=clock, home=home, away=away, home_fouls=0, away_fouls=0, conf=1.0)
        for i in range(seconds)]


def test_a_minutes_misread_that_lasts_a_while_does_not_end_the_quarter():
    """At Marcos de Niza the 8:00 shows for a few seconds before the tip, then the 7 reads as 8 for the
    first seconds the clock runs (8:51, 8:50, 8:49, then 7:49). Later a whole minute reads 8:xx for 7:xx.
    Neither is a new clock: the seconds carry on counting."""
    tl = _static(0, 1, 540, 300)                                     # pregame: 9:00 held on the board
    q = _static(300, 2, 480, 10)                                     # 8:00 held until the tip
    q += countdown(310, 2, 480, 480)                                 # the quarter
    for r in q[10:14]:
        r.clock_s += 60                                              # 7:5x reads as 8:5x right after the tip
    for r in q[130:190]:
        r.clock_s += 60                                              # and a whole minute of 6:xx reads as 7:xx
    tl += _score(q, 40, start_t=310)
    phases = game_phases(tl, quarter_length_s=480.0)
    assert len(phases.live) == 1
    assert not is_live(phases, 200) and is_live(phases, 320) and is_live(phases, 450) and is_live(phases, 700)


def test_a_timeout_countdown_on_the_game_clock_does_not_end_the_quarter():
    """The game clock shows the 30-second timeout countdown (0:30 down to 0:10) and then returns to
    where the quarter stood. The quarter continues."""
    q = countdown(0, 1, 480, 480)
    for i, r in enumerate(q[200:221]):
        r.clock_s = 30.0 - i                                         # a timeout at 4:40: the clock shows 0:30 .. 0:10
    for r in q[221:260]:
        r.clock_s = 280.0                                            # back to 4:40, held through the rest of the timeout
    for r in q[260:]:
        r.clock_s += 59                                              # then the countdown resumes from 4:40
    tl = _score(q, 40)
    phases = game_phases(tl, quarter_length_s=480.0)
    assert len(phases.live) == 1 and is_live(phases, 210) and is_live(phases, 240) and is_live(phases, 400)


def test_a_quarter_whose_first_minute_reads_a_minute_high_still_starts_on_time():
    """The board is set to 8:00 but reads 9:00, and the whole first minute reads a minute high before the
    digit settles. Most of the quarter reads right, so the quarter is 8:00 and starts where the clock was set."""
    tl = _static(0, 1, 60, 80, 20, 18)                               # the break clock held at 1:00
    q = countdown(80, 2, 480, 480, 20, 18)
    for r in q[:60]:
        r.clock_s += 60
    tl += _score(q, 40, start_t=80)
    phases = game_phases(tl, quarter_length_s=480.0)
    assert len(phases.live) == 1 and phases.live[0][0] <= 81 and not is_live(phases, 50) and is_live(phases, 90)


def test_the_tens_digit_of_the_halftime_clock_dropping_out_is_not_a_quarter():
    """Halftime counts down from 15:00; the tens digit drops out now and then, so 14:06 reads 4:06 for
    a stretch. No read of it is a quarter: nothing scores, and the countdown is one clock."""
    half = countdown(0, 2, 900, 900, 30, 28)
    for r in half[50:120]:
        r.clock_s -= 600
    for r in half[400:470]:
        r.clock_s -= 600                                             # 8:20 .. 7:10 reads as the quarter length would
    q = _score(countdown(900, 3, 480, 480, 30, 28), 40, start_t=900)
    phases = game_phases(half + q, quarter_length_s=480.0)
    assert len(phases.live) == 1 and not is_live(phases, 420) and is_live(phases, 1000)


def test_quarters_are_numbered_by_the_board_period_and_overtime_is_half_a_quarter():
    tl = []
    for k, period in enumerate((1, 2, 3, 4)):
        q = countdown(k * 600, period, 480, 480, 20 * k, 20 * k)
        tl += _score(q, 60, start_t=k * 600)
        tl += _static(k * 600 + 480, period, 0, 120, 20 * k + 14, 20 * k + 14)   # 0:00 held, final score of the quarter
    ot = countdown(2400, 5, 240, 240, 80, 80)
    tl += _score(ot, 60, start_t=2400, home=False)
    phases = game_phases(tl, quarter_length_s=480.0)
    assert list(phases.quarters) == [1, 2, 3, 4, 5] and phases.live == list(phases.quarters.values())
    assert is_live(phases, 2500) and not is_live(phases, 2300)


def test_a_quarter_whose_last_minute_went_unread_still_covers_it():
    """At Saguaro the board became unreadable with 1:00 on the clock and stayed so until the break clock
    was set three minutes later. The minute was played: the quarter runs on, up to three times the time
    left, and never past the next clock setting."""
    q = _score(countdown(0, 1, 480, 420), 40)                         # reads stop at 1:00
    tl = q + _static(600, 2, 60, 30, 20, 20)                         # the break clock, set 180 s later
    tl += _score(countdown(660, 2, 480, 480, 20, 20), 40, start_t=660)
    phases = game_phases(tl, quarter_length_s=480.0)
    assert len(phases.live) == 2
    assert is_live(phases, 500) and is_live(phases, 595) and not is_live(phases, 605)
    short = _score(countdown(0, 1, 480, 470), 40) + _static(900, 2, 60, 30, 24, 24)   # last read 0:11 at 469, nothing set for 15 min
    assert not is_live(game_phases(short, quarter_length_s=480.0), 469 + 11 * 3 + END_GRACE_S + 1)

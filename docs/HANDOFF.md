# Handoff: basketball film pipeline

Written 2026-10-01, updated 2026-10-01 late (made-or-missed classifier in service), for the next agent. Read this before touching anything. The design spec
(`docs/superpowers/specs/2026-09-30-film-pipeline-design.md`) says what the system is meant to become;
this file says where it actually is, what was learned running it on real film, and what to do next.

## Start here

**Your job is the no-scoreboard path**: making the pipeline work on film that has no burned-in Hudl
scoreboard (the NFHS camera and iPad recordings). Garrett confirmed this priority on 2026-10-01. Do not
process more Hudl games first; the pipeline already handles those.

Do these in order. Details for each are under "Next steps" below.

1. Done: Garrett reviewed the Cactus Shadows game (121 decisions: 107 confirmed, 14 rejected).
2. Done: the made-or-missed classifier (`rimclf`) is past its quality gate and in service as `rimclf@v4` (on `ball@v5`).
   It reads the ball's path after the rim, not pixels. It is right about three times in four, so the next
   gains come from ball labels just after the shot (see "What does not work yet").
3. Train a rim detector that works from any camera angle (the current one is a template matcher that
   only fits the Hudl view of the Arcadia gym).
4. Add a no-scoreboard profile with manual quarter marks, and run `JV vs Deer Valley JV - Angle 1.mp4` as
   the first test, then `Prescott vs Arcadia .mp4`, then `RKDA @ MDN 1:24.mp4`.

Before any run longer than about a minute, tell Garrett what it is and how long it takes, and wait for
his go-ahead: it slows his laptop.

## The project in one paragraph

Garrett Lamb coaches JV basketball at Arcadia High School (the Titans). He charts games by hand in a
shot tracker (`~/coaching-sim-web/public/shot-tracker/`) and wants film to do it for him, Hudl-Assist
style: upload a game, get shots (then lineups, plus-minus, a box score), confirm them on a fast review
page. The machine proposes, the coach confirms, and every confirmation becomes training data. First
game he wants it working for: late November 2026. Everything runs locally on his M3 / 8 GB MacBook; no
rented GPU. He may buy a Mac mini later but has decided to prove it on the laptop first.

## State on 2026-10-01

- `main`, 243 tests passing, **no git remote yet** (he will supply an SSH URL).
- Rung 1 of the spec is built and has run end to end on two real games.
- Two games processed and reviewed by Garrett (Cactus Shadows: the 125 ball and scoreboard cards; its 108 motion cards are not reviewed).
- Three more films are on disk that the pipeline **cannot run yet** because they have no burned-in scoreboard.

### What works, with measured numbers (JV vs Marcos, the reviewed game)

| Piece | How | Result |
|---|---|---|
| Camera profile | Correlate the bottom-left corner with a template of the Hudl overlay | 0.95+ on Hudl frames, under 0.13 on others |
| Scoreboard | Seven-segment decode of the burned-in Hudl overlay, once a second, then plausibility cleaning | 5,309 reads in a 90-minute game; score, clock, period, team fouls |
| Rims | Template match of the top of each backboard (`film/perceive/rim_templates.py`); left = id 1, right = id 2 | Rim found in 52% of frames (the camera is elsewhere the rest of the time) |
| Ball | YOLO11n trained on Garrett's 82 boxes, run on the 320 px rim crop enlarged to 640 (`ball@v4`) | 9,129 detections; 139 ball arrivals in live play |
| Shot events | Ball crossing the rim line, with a peak-confidence gate; motion peaks as fallback; scoreboard-only events for baskets nothing else explains | 139 confirmed shots (66 made, 73 missed); 91 of 119 final points accounted for |
| Made or missed | The scoreboard: did one team's score rise by 1-3 in the seconds after the arrival | Agrees with Garrett on 94% of confirmed shots |
| Made or missed without a scoreboard | `rimclf@v4`: the ball detector at 15 fps for 1.4 s after each arrival, six features of the ball's path, a linear model (`film/interpret/ballpath.py`, `film/train/rimclf.py`) | 0.748 on three classes trained on one game and tested on the other (always guessing scores 0.488); 0.785 on made versus missed (161 of 205). With `ball@v4` it was 0.708 and 0.767 |
| Warmups | `film/interpret/phases.py`: only stretches whose clock starts at the quarter length are live | Flagged all 43 cards he skipped as warmups, none he confirmed |
| Review | Local page, keys Enter / M / T / X / U / L / R / F / J / K | 266 of 267 live cards decided |

Card quality by source on that game: ball cards 83% real (115 of 138), scoreboard-only 9 of 9, motion
cards 13% real (15 of 118). Motion is kept only because it still catches some shots the ball detector misses.

Speed on the laptop: Perceive runs about 34 frames/s, so 26-30 minutes for a 90-100 minute game.
Cutting ~300 clips takes about 2 minutes. Training the ball detector at 640 px took 16 minutes.

### What does not work yet

- **The made-or-missed classifier is in service but only about 75% right.** The scoreboard is right about
  95% of the time where it has a read, and still wins when the two disagree (`clf-scoreboard-disagree`).
  On the shots with no scoreboard read the classifier's call matched Garrett on 34 of 46 across the two
  games (in-sample). What limits it: the ball detector loses the ball in about a third of makes by 0.6 s
  after arrival. More ball labels from the second after a shot are the direct fix: the first such batch
  (`film prelabel ball <id> --after-shots`, 87 boxes and 80 hard no-ball crops) moved it from 0.708 to 0.748.
  The detector has now seen labels from both games, so leave-one-game-out is no longer fully blind for it.
  Garrett presses K for "no ball" out of habit; a skipped crop is deleted, but its id carries the time and
  rim, so it can be regenerated and saved empty (done once by hand for the crops `ball@v4` wrongly boxed).
  What was tried and failed, so nobody repeats it: an image classifier (ResNet18) on rim montages, four
  input designs (the old 96 px montage, tight full-resolution crops, a longer window, frame differences),
  each trained on one game and tested on the other: 0.51 to 0.61 on three classes and 0.55 to 0.60 on made
  versus missed, where always answering "missed" scores 0.575. Two games are too few clips for pixels.
- **On Hudl film the "rim" box is the top of the backboard, not the hoop.** The hoop sits about 0.4 rim
  widths to one side of the box centre (right for rim 1, left for rim 2). `hoop_offset` measures this per
  game and rim from where confident balls arrive; it needs at least 5 arrivals or it falls back to the box
  centre. A trained rim detector on other cameras will box the real rim, and the classifier's thresholds
  (in rim-box widths) will need retraining on labeled film from that camera.
- **Any film without the Hudl overlay.** `film run` refuses an `unknown` profile on purpose. No trained rim
  detector exists (YOLO-World cannot find these rims at any prompt or size: it boxes the ball instead),
  and there is no manual quarter marking. Made-or-missed now has a source (`rimclf@v3`), trained on the Hudl view only.
- **Missed shots are under-detected.** The ball detector catches 29 of the first 53 confirmed shots by
  itself; makes are rescued by the scoreboard, misses are not.
- Shot locations, teams by jersey colour, tracker export, lineups, fouls by player, box score: not started
  (rungs 2-5 of the spec).

## Films on disk

All in `~/Downloads` (the `.mp4` copies are fully local) and `~/Desktop/Full Games` (some `.mov` copies
there are iCloud placeholders with 0 bytes on disk; check with `du -k` before using them).

| File | Camera | Status |
|---|---|---|
| `~/Downloads/JV vs Marcos - Tactical.mp4` | Hudl Focus, home gym, burned-in scoreboard | Processed and reviewed: `games/2025-02-05-jv-vs-marcos-tactical` |
| `~/Downloads/Arcadia High School vs Cactus Shadows Boys' JuniorVarsity Basketball - Tactical.mp4` | Same | Processed, **233 live cards waiting for Garrett's review** (125 ball/scoreboard first, 108 motion after) |
| `~/Downloads/JV vs Deer Valley JV - Angle 1.mp4` | NFHS Network auto-camera, home gym; pans and zooms, no overlay | Refused: profile unknown, templates find no rims |
| `~/Desktop/Full Games/Prescott vs Arcadia .mp4` (space before `.mp4`) | iPad, a few rows up in the bleachers, home gym, opposite side from the Hudl camera; 1080p, sharp; wall scoreboard in frame at one end | Not processed. The easier iPad case |
| `~/Desktop/Full Games/RKDA @ MDN 1:24.mp4` | iPad at court level at Marcos de Niza; 720p, motion blur on pans; wall scoreboard in frame at one end | Not processed. The hard iPad case |
| `~/Downloads/State Championship.mp4` | NFHS broadcast with a score bug, 720p | Not examined beyond one frame |
| `~/Desktop/Full Games/Arcadia Semi Final .mov` | Not examined | |

Game ids carry the **download** date (2025-02-05), not the real game date; Garrett has not supplied dates.

## Data on disk (all git-ignored)

- `games/<id>/`: `manifest.json`, `tracks.parquet`, `motion.parquet`, `scoreboard.jsonl`, `events.json`,
  `clips/`, `review.json`. **`review.json` is Garrett's work and is irreplaceable**; a copy of his first 175
  decisions is in `games/_backups/`. Interpret carries decisions onto re-detected shots
  (`carry_over_decisions`), and nothing ever deletes a decision.
- `models/ball/v5` is in service: 169 boxes plus 83 no-ball crops, 38 minutes to train on the laptop (v4: 82 boxes;
  v1, v2 are older; v3 is an empty folder from an interrupted run). `tracks.parquet` of both games was still
  written by `ball@v4`: a `film perceive --force` (half an hour a game) would let v5 find arrivals too.
  `models/rimclf/v4` is in service (ball-path, paths read by `ball@v5`; v3 was the same on `ball@v4`); `v1` (image classifier) and `v2` (ball-path, one game) hold
  `not-good-enough.pt` and are ignored by the registry. `models/base/` holds the
  YOLO-World weights; `yolo11n.pt` re-downloads there on the next detector training.
- `datasets/ball`: 252 crops (169 with a ball box, 83 saved empty as "no ball here"). `datasets/rimclf`: 449 labeled montages from both games (99 made, 131 missed,
  219 none), each with its ball path in `datasets/rimclf/paths/<game_id>--<event_id>.json`. The montage is
  what the coach sees in Label clips; the path is what trains. `datasets/rimclf-montage-v1` and
  `labels/rimclf-montage-v1` are the old image-only dataset, set aside, safe to delete.
- `games/<id>/ballpaths.json`: the ball path of every arrival, read once per ball-detector version.
- `games/_backups/pre-rimclf-v3/`: events and reviews of both games from before the classifier went in.
- `labels/rimclf/pending`: 315 montages (warmup cards and unreviewed motion cards; optional labels). `labels/rim/pending`: 10 stale test frames.

## Update 2026-10-02: the first no-scoreboard game (Deer Valley, NFHS camera)

- **Rim detector for cameras without templates:** `rim@v1`, trained on 150 Deer Valley frames Garrett
  labeled with **backboard** boxes (the rim is a few pixels in a wide shot). `rim_from_backboard`
  (`film/perceive/rims.py`) turns a backboard into the rim reference strip. 0.987 mAP50 on held-out frames.
  Profiles with templates (Hudl) keep their templates first; `FILM_RIM_DETECTOR=trained` reverses that.
- **Crops scale with the rim** (`CROP_WIDTHS = 3.75`, `film/perceive/ball.py`), shrunk to 320 px, so a zoomed
  camera's ball is the size the detector learned. Hudl crops are unchanged (under 320 px).
- **Quarter marks** (`film/interpret/quarters.py`, Quarters tab, `[` and `]`): Garrett marked Deer Valley's four
  quarters; they set warmups and quarter numbers. No clock without a scoreboard.
- **Deer Valley run:** 54 minutes on the laptop (rim network on every frame, 23 frames/s). 793 cards, 423 warmup.
  Garrett reviewed the 196 ball cards: **92 real (47%)**; made/missed was a coin toss.
- **Why made/missed failed there:** the ball detector barely saw the ball on this camera (confident in ~1 frame
  in 10), so every path looked alike. A Deer Valley after-shot ball batch (141 boxes, 15 empty) and `ball@v6`
  fixed the seeing; `rimclf@v6` is in service: 0.666 leave-one-game-out over three games; on Deer Valley
  without training on it 0.54 (guessing 0.61), made vs missed 0.60, flags 77 of 144 fake cards while keeping
  81 of 92 real shots. Hudl games: 0.745 / 0.723 (were 0.771 / 0.737); the scoreboard still decides there.
  Cards the classifier calls empty now sort with the motion cards.
- **Next:** another NFHS game reviewed the same way (one game of a camera is not enough), then the iPad
  angle (Prescott first: backboard labels, then shots). Parked: clip length; box-score stats including
  turnovers, steals and blocks.
- Label tabs: `D` skips (next to `S`); `S` with no box saves "no ball here". Garrett reads K/D as "can't tell".

## What Garrett has asked for (requirements gathered in conversation)

1. **Full box score eventually**, as automated as possible; he accepts reviewing some of it.
   He named **turnovers, steals and blocks** explicitly (2026-10-02): they must be in the box score, not
   only shots, rebounds and fouls. A block is tied to a shot (a shot card can carry "blocked by #"); a
   turnover, and a steal when one caused it, come from possession changes, which need the ball tracked
   across the court and players identified (the box-score rung).
2. **Jersey number + team is the identity key.** Names optional. When the machine is unsure of a player it
   should show him crops and let him type the number; worst case give him timestamps.
3. **Substitutions, lineups, plus-minus, lineup efficiency.**
4. **Clips should end at the outcome** (make, or the rebound after a miss) so he can watch the play
   develop; later, start each clip at the start of the possession. Today: 8 s before, 3 s after.
5. **Fouls:** when, which team, which player. Team fouls and timeouts are already in `scoreboard.jsonl`.
   The overlay's PLAYER panel is dark at Arcadia, so the player needs a foul card where he types the number.
6. **Many games will have no burned-in scoreboard** (NFHS camera, iPad). This is a main path, not an
   add-on. He wants the pipeline "prepared for all scenarios".
7. **Tracker uses the high-school three-point arc** (19.75 ft, no corner segment). Spec section 9.
8. Local compute only. No rented GPU.

## How to work with Garrett

- Explain tradeoffs in plain language first, then let him choose. Lead with evidence (tables, measured
  numbers, an image) rather than assertions.
- Give exact commands and exact keys. He is not a programmer by trade and asks "what am I doing here"
  when a step is not spelled out.
- **Ask before any run that takes more than about a minute.** Training or a full-game pass slows his
  laptop and he usually has other sessions open. Say what it is and how long, then wait. Quote long
  estimates: training at 640 px took 16 minutes against an estimate of 3-5.
- Never kill processes from his other projects (for example `Kids League Website` checks).
- When he says a page "isn't opening", start the review server in the background and `open` the URL for
  him: `open "http://127.0.0.1:8765/#ball"` (`#shots`, `#rim`, `#ball`, `#rimclf` select the tab).
- Pipeline commands use no Claude tokens; he asked.

## Next steps, in the order I recommend

The question he asked last: prioritise the no-scoreboard path, or feed a third scoreboard game first?
**Recommendation: build the no-scoreboard path next, and do not process a third Hudl game yet.** The one
thing that path needs from more Hudl film is classifier data, and the Cactus Shadows game already
supplies a second batch as soon as he reviews it.

1. Done: Cactus Shadows reviewed.
2. Done: `rimclf` clears the gate (see above). `film train rimclf` now scores by leave-one-game-out when the
   dataset holds two or more games, and that score is what the gate reads.
3. **A trained rim detector that is not tied to one camera.** On Hudl film the template matcher can label
   rims automatically (free training data for that view); for the NFHS and iPad views Garrett labels
   about 150 frames each in the Label rim tab. Then `film train rim`. Set `FILM_RIM_DETECTOR=template` on
   Hudl film so the trained model does not displace the templates there.
4. **A `no-scoreboard` profile:** manual quarter marks in review setup (spec section 8) for quarter, clock
   and warmup time; score summed from confirmed shots; Interpret using the classifier alone. Test on
   `JV vs Deer Valley JV - Angle 1.mp4` first (same gym, same ball), then `Prescott vs Arcadia .mp4`.
5. **A wall-scoreboard reader** for film where the gym's scoreboard is in frame at one end (both iPad
   games). Intermittent reads are enough to check makes and track the period.
6. **More ball labels.** `film prelabel ball <id>` now samples densely around confirmed shots and never
   re-queues a skipped crop. A batch from the 86 shots he confirmed last, plus the Cactus Shadows shots,
   should lift recall on misses. Consider an "N = no ball here" key that saves background negatives.
   Retire motion arrivals once ball recall makes them redundant.
7. **Clip cutting (Garrett, 2026-10-02: "we have to find a way to improve the clip cutting / individual
   clip length", parked for later).** Today every clip is a fixed 8 s before the arrival and 3 s after. On the
   zoomed NFHS camera this shows a lot of dead time and makes neighbouring cards look like duplicates. Ask him
   what bothers him most before changing it; requirement 4 above (end at the outcome, later start at the
   possession) is the direction he gave earlier.
7. **Rung 2 of the spec** (shot locations, teams by jersey colour, tracker export, high-school arc), then
   lineups and fouls. Each rung gets its own plan under `docs/superpowers/plans/`.

Smaller debts are listed in `docs/superpowers/plans/2026-09-30-rung-1-execution-ledger.md` (lines marked
`minor (deferred)` and `Parked`).

## Gotchas that cost time

- `uv` lives in `~/.local/bin`; run `export PATH="$HOME/.local/bin:$PATH"` first. There is no Homebrew.
- Python's default certificate store fails on this Mac; `film` sets `SSL_CERT_FILE` to certifi's bundle.
  A bare `python` download needs the same.
- PyAV and OpenCV both bundle libavdevice, so every run prints an objc "class implemented in both"
  warning. It is harmless for file decoding; filter it out of logs.
- The ball detector must run at the size it was trained at; `load_detector` reads `imgsz` from the model's
  manifest. `IMGSZ["ball"]` is 640.
- `film interpret` now reads ball paths from the film when a classifier is in service: about 2 minutes on a
  game's first run (ask Garrett first), seconds afterwards (`ballpaths.json`). `--no-classifier` skips it.
- Games perceived before 2026-10-01 carry a blank box on the first rim row of each rim (a smoothing bug, fixed);
  `rim_follower` and `rim_box_at` skip such rows.
- A `film run --limit-s N` smoke test does not count as a finished Perceive; a later full run redoes it.
  `film run` otherwise skips finished stages; `--force` re-runs them.
- Re-running Interpret changes event ids when arrival times shift. Decisions follow their shots, but back
  up `review.json` before any experiment that rewrites `events.json`.
- Changing clip windows needs `film clips <id> --force`.
- Subagents and other models inject their own `Co-Authored-By` trailer; use whatever the current session's
  attribution rule says and check with `git log -1 --format=%B`.
- The rung-1 work was done with the superpowers workflow (brainstorm, spec, plan, subagent-driven
  execution with per-task review). The plan and ledger are in `docs/superpowers/plans/`.

## Two checks worth recreating

They lived in a session scratchpad and are gone; both are a page of code.

- **Recall on confirmed shots.** For each confirmed event, decode a few seconds around it, run the rim and
  ball detectors, call `find_ball_arrivals`, and count events with at least one arrival. This is how
  `ball@v2` (21 of 53) and `ball@v4` (32 of 53) were compared in two minutes instead of a full-game run.
- **Rule sweep.** Load `tracks.parquet` and the decisions, recompute arrivals under different rule
  parameters, and report hits on confirmed shots, hits on rejected cards, and arrivals near no card. This
  is how the peak-confidence gate was chosen.

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

- `main`, 280 tests passing, remote `origin` = `git@github.com:garrettjlamb-collab/GARRASSIST.git` (private, since 2026-10-12; push after every commit).
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

All in `~/Downloads` (the `.mp4` copies are fully local) and `~/Desktop/Full Games`, which syncs to iCloud
Drive. Most films there are in iCloud only, with no copy on the Mac: `film doctor` lists which.

### Films may be cloud-only (built 2026-10-09)

The laptop's disk is too small to keep every film, so a job pulls its film from iCloud when it needs it
and puts it back when it ends (`film/common/cloud.py`; spec in
`docs/superpowers/specs/2026-10-09-cloud-film-design.md`).

- **Never open `m.source` directly.** A cloud-only file opened that way is downloaded by macOS with no
  space check and never goes back. Any script that reads a film, including a one-off, does this:

  ```python
  from film.common.cloud import game_film, local_source

  with local_source(game_film(game_id)) as film:
      frame = first_frame_at(film, t, fps)
  ```

- Only a film the pipeline pulled goes back. One Garrett keeps on the Mac is never offloaded.
- A film is held until the last job using it ends, across processes, so two sessions can share one.
- A pull is refused when the disk does not have the film's size plus 5 GB free. Each film is 0.6 to 5.7 GB.
- macOS will not offload a file that has not finished uploading, and an offload it accepts does not always
  take. Both are reported in one line (`... is still on the Mac`) and the next job on that film tries again.
- The review cards play pre-cut clips and need no film. The Quarters tab and the L key pull the full film
  and the server puts it back after 30 minutes with nobody watching, or when it stops.
- The films in `~/Downloads` are not in iCloud and are left alone.

| File | Camera | Status |
|---|---|---|
| `~/Downloads/JV vs Marcos - Tactical.mp4` | Hudl Focus, home gym, burned-in scoreboard | Processed and reviewed: `games/2025-02-05-jv-vs-marcos-tactical` |
| `~/Downloads/Arcadia High School vs Cactus Shadows Boys' JuniorVarsity Basketball - Tactical.mp4` | Same | Processed, **233 live cards waiting for Garrett's review** (125 ball/scoreboard first, 108 motion after) |
| `~/Downloads/JV vs Deer Valley JV - Angle 1.mp4` | NFHS Network auto-camera; pans and zooms, no overlay. Not Garrett's gym (he said so 2026-10-02; the earlier "home gym" here was a guess), and he has no other NFHS film from it | Processed and reviewed (196 ball cards): `games/2025-02-05-jv-vs-deer-valley-jv-angle-1` |
| `~/Desktop/Full Games/Prescott vs Arcadia .mp4` (space before `.mp4`) | iPad, a few rows up in the bleachers, home gym, opposite side from the Hudl camera; 1080p, sharp; wall scoreboard in frame at one end | Not processed. The easier iPad case |
| `~/Desktop/Full Games/RKDA @ MDN 1:24.mp4` | iPad at court level at Marcos de Niza; 720p, motion blur on pans; wall scoreboard in frame at one end | Not processed. The hard iPad case. No Hudl copy exists (Garrett, 2026-10-02) |
| `~/Desktop/Full Games/Combs vs MDN - Angle 1.mp4` | iPad at court level in the Combs gym (not NFHS despite "Angle 1"; Garrett confirmed). 1080p, motion blur, wall scoreboard in frame, side-wall practice hoops in view | Ingested as `2026-10-01-combs-vs-mdn-angle-1` (handheld); 100 frames queued for backboard labels |
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
- **Next:** the iPad angle (Prescott: backboard labels, then shots). Garrett has no more NFHS film from that gym;
  a second NFHS game from anywhere would still help the classifier. Parked: clip length; box-score stats including
  turnovers, steals and blocks.
- Label tabs: `D` skips (next to `S`); `S` with no box saves "no ball here". Garrett reads K/D as "can't tell".

## Update 2026-10-03: the first iPad game (Prescott), and baskets by side

- **`rim@v2`**: backboard detector retrained on Deer Valley + Prescott frames (276; `rim@v1` had already boxed 8 of
  9 Prescott backboards unprompted). The run hit the session's 2-hour background limit at round 45 of 50 because
  the laptop was swapping; `best.pt` (round 13, mAP50 0.995) was registered by hand, noted in its manifest.
  **Long jobs on this laptop: check `sysctl vm.swapusage` first, and keep one job under 2 hours.**
- **Prescott** (`games/2026-10-01-prescott-vs-arcadia`, profile `handheld`, 25 minutes to process): 175 cards.
  Garrett reviewed 119: strong cards **90 of 111 real (81%)**, as good as Hudl. Made/missed was a coin toss.
  The film holds game time only (the iPad was stopped at breaks), so there are no warmups.
- **Baskets by side** (`basket_sides`, `film/perceive/rims.py`): a panning camera gives a basket a new rim id each
  time it returns (Deer Valley 198 ids, Prescott 90), so the per-id hoop offset was mostly unmeasured. A track is
  the left basket when it lives in the left half of the frame; offsets are pooled per side. They agree across
  cameras: about +0.5 rim widths left, -0.5 right.
- **`rimclf@v8`** in service: 0.624 leave-one-game-out over four games (guessing 0.505). Made versus missed with
  the game held out: Cactus Shadows ~0.81, Marcos ~0.72, Prescott 0.66 (was 0.54 before sides), **Deer Valley
  0.55, still chance**. Deer Valley is the open problem: tight zoom, 47% card precision, no signal in the path
  features even when trained on its own other half. Look at its clips before more labels.

## Update 2026-10-05: a second gym on iPad (RKDA @ MDN), and what a new gym costs

- **A new gym needs labels twice, about 15 minutes each from Garrett.** `rim@v2` (Arcadia only) found no backboard
  at Marcos de Niza; `ball@v6` boxed its yellow banners, pom-poms and scoreboard digits (460 cards for a 57-minute
  game). After 121 RKDA backboard frames (+83 Combs) -> `rim@v3` (0.989), and 75 RKDA ball boxes + 104 empty
  crops -> `ball@v7` (0.866): **175 cards, 121 of 127 strong cards real (95%), made/missed right on 82%**.
- `film prelabel ball <id> --no-proposals` queues bare crops for such a gym. Garrett presses D/K for "no ball"
  whatever the instructions say: ask him, then regenerate the skipped ids and save the ones the detector fired on
  as empty (done by hand twice; worth making a command).
- A temporal "fixed spot" filter for false ball sightings was tried and dropped (460 -> ~420 cards only).
- **`rimclf@v9`** in service, paths read by `ball@v7` on all five games: 0.684 leave-one-game-out (guessing 0.467).
  Made versus missed, game held out: Cactus Shadows 0.82, RKDA 0.78, Marcos 0.75, Prescott 0.73, **Deer Valley
  0.58 (still the open problem)**.
- Skipped frames can be re-queued from their ids (rim: `<game>_<t*10>`; done once for 58 accidental skips).
- **Combs** (`2026-10-01-combs-vs-mdn-angle-1`, iPad, third gym): backboards labeled (in `rim@v3`), not processed.
  Next: process it, look at the card count, then a ball batch for that gym if it is inflated.
- `tracks.parquet` of Marcos, Cactus Shadows, Deer Valley and Prescott still come from older ball detectors
  (v4 to v6); a `film perceive` of each (25 to 55 minutes) would let `ball@v7` find their arrivals.

## Update 2026-10-05 (later): Combs, a third gym with no ball labels

- **Combs** (`2026-10-01-combs-vs-mdn-angle-1`, iPad, 31 minutes to process) needed backboard labels only:
  `ball@v7` had never seen the gym and still followed the real ball. 305 cards; Garrett reviewed 190:
  **137 of 188 strong cards real (73%)**, made/missed right on 74% with a classifier that had not seen the game.
- **`rimclf@v10`** in service, six games: 0.653 leave-one-game-out (guessing 0.454). Made versus missed, game held
  out: Cactus Shadows 0.82, RKDA 0.77, Prescott 0.74, Combs 0.74, Marcos 0.74, Deer Valley 0.59.
  Its "none" class does not transfer to Combs (18 of 91 fake cards caught, against 30 of 47 on RKDA).
- Six games reviewed: 2 Hudl, 1 NFHS, 3 iPad in 3 gyms. Garrett's decisions: Marcos 284, Cactus Shadows 121,
  Deer Valley 196, Prescott 119, RKDA 128, Combs 190.

## Update 2026-10-06: the picture classifier works now, and it improves with more games

Measured with `experiments/rimclf_images.py` (in the repo; sheets in `datasets/rimclf-images`), every game held
out in turn, 678 real shots and 564 non-shots across six games:

| Made versus missed, game held out | Ball path (`rimclf@v10`) | Pictures (ResNet18 per frame, 16 frames, pooled) | Averaged |
|---|---|---|---|
| Cactus Shadows | 0.82 | 0.84 | 0.87 |
| Marcos | 0.74 | 0.72 | 0.75 |
| Deer Valley | 0.59 | 0.79 | 0.74 |
| Prescott | 0.74 | 0.86 | 0.88 |
| RKDA | 0.77 | 0.75 | 0.81 |
| Combs | 0.74 | 0.91 | 0.91 |
| All | 0.74 | 0.81 | 0.83 |

- **When the two agree with averaged confidence >= 0.75: 57% of real shots, 95% right** (>= 0.65: 67%, 91%).
  When they disagree it is a coin toss. This is the no-scoreboard version of "scoreboard and classifier agree".
- **The path classifier is flat with more games** (0.72 on one game, 0.73 on five: six features, saturated).
  **The picture classifier climbs:** Combs held out 0.53 / 0.67 / 0.78 / 0.86 / 0.91 on 1 to 5 games; RKDA
  0.60 / 0.63 / 0.60 / 0.74 / 0.75. Each reviewed game now makes the next one better.
- The same picture approach failed on two games in September (see above): it needed about 1,000 clips, a crop
  1.6 rim widths across centred under the measured hoop, and every frame through the network separately.
- **In the pipeline since 2026-10-07: `picclf@v1`** (`film/interpret/rimview.py`, `film/train/picclf.py`,
  `film/interpret/fuse.py`). `film prelabel rimclf <id>` also cuts each item's rim view
  (`datasets/rimclf/views`); `film train picclf [--hold <game_id>]` scores on one held-out game (default: the
  one labeled last; Combs gave 0.822 three-class, 0.906 made versus missed) and registers a model fit on all
  games: 17 minutes on the laptop for 1,242 views. Interpret keeps each arrival's picture scores in
  `games/<id>/picviews.json`.
- **Card confidence without a scoreboard** (`AGREEMENT_CONFIDENCE`, `film/interpret/events.py`): strong
  agreement 0.92, fair 0.8, split 0.4 with the flag `classifiers-disagree`. "Strong" needs conviction >= 0.75
  and an averaged "none" <= 0.25; held out, such a card is a real shot with the right call 90% of the time
  (348 cards, 48% of real shots). Motion-only cards stay capped at 0.5.
- On the six reviewed games after the re-run: 484 reviewed cards at >= 0.9, 480 real and right, covering 69% of
  real shots. **That is in-sample** (the model in service trained on these games); expect about 90% and about
  half the shots on a new game. The first new game is the real test: review it in full and compare.
- Cost: about 8 minutes per held-out game on the laptop (8 epochs, ~1,000 clips).

## 2026-10-10: ten more games reviewed (sixteen in all)

Garrett reviewed the 2026-10-08 and 2026-10-09 games with the "Confirm all above 0.9" button, except two he did
in full first (@Cactus Shadows iPad, MDN vs St. Mary's). Across the hand-reviewed strong cards: **446 of 496
real (90%), made/missed right on 388 of 446 (87%)**; 670 cards taken by the button; 1,111 confirmed shots on
these ten games. Honest button check (games unseen by `picclf@v1`): St. Mary's 42/44 right, @CS iPad 38/45.
Left: MDN vs Cactus Shadows (iPad, 2026-10-08) not started; St. Mary's 18 strong cards. Reviews backed up in
`games/_backups/<date>-reviews/`. Fixed on the way: a clockless board's quarters now come from runs of scoring
split at 4-minute gaps (halftime was being counted as play; the period digit misreads during breaks), the
Quarters tab arrows step exactly 5 s (the player had focus and added its own skips), and the queue sends
confirmed cards to the back and lands on the first card to review. **Next: rebuild the classifier datasets from
all sixteen games (`film prelabel rimclf <id>` each, ~3 min a game) and retrain `rimclf` and `picclf`.**

## 2026-10-09 night: retrained on sixteen games

`rimclf@v11` (0.684 leave-one-game-out, guessing 0.379) and `picclf@v2` (0.808 on St. Mary's held out, 0.899
made versus missed) are in service; every game re-interpreted with them and with the repeat-crossing rule
(`REPEAT_WINDOW_S`, `film/interpret/arrivals.py`). 2,010 decisions kept, 31 moved, 73 orphaned (mostly the
merged duplicates). MDN vs Cactus Shadows (iPad): 373 -> 317 cards, confirmed pairs under 3 s 47 -> 9; Garrett
still has 72 strong cards there. In-sample, safe cards are 1,113 of 1,117 right covering 69% of confirmed shots;
the honest figure stays the held-out one (84-95%). Datasets: `datasets/rimclf` now holds about 2,900 items from
sixteen games. Next: the five home Hudl games he plans to download; then the wall-scoreboard reader, clip
cutting, rung 2.

## Home gym is Marcos de Niza (Garrett, 2026-10-12)

The five MDN Hudl games are his home games. So the `hudl-mdn` skin is the one that matters most: reading its
**clock** (not done; its font defeats the segment zones) would give warmups and the game clock on every home
game without quarter marks. Arcadia is history. Build order agreed: MDN clock reader -> wall-scoreboard reader
for iPad film -> clip cutting -> shot locations and the tracker export -> box score.

## Naming (2026-10-09)
Game ids are folder names and keys into datasets, paths and views: **never rename a game folder**. A game now
has a `title` for people (`film title <id> "..."`, shown on the review page); ids from new downloads no longer
carry a "NEW " prefix. Titles currently carry Garrett's to-do step ("1 · Mark quarters, then review IN FULL ·
..."); reset them to plain names once a game is reviewed. Dates in ids are still download dates.

## 2026-10-09: six more films; the Marcos de Niza Hudl skin is readable

- `~/Desktop/Full Games/NEW @Cactus Shadows - Angle 1.mp4` (iPad, CS gym, backboard 8 of 8 unprompted) ->
  `2026-10-09-new-cactus-shadows-angle-1`, profile handheld.
- Five `NEW Marcos de Niza High School vs <St. Mary's | Cactus Shadows | Arcadia | Mesquite | Saguaro> ...
  Tactical.mp4`: Hudl Focus at MDN, 150-minute films, the game in roughly the first 75 minutes, backboards tiny.
  **Profile `hudl-mdn`** (`film/perceive/profiles/hudl_mdn.json`): home, away and period from the school's LED-dot
  skin (threshold 0.15, cells measured from 36 frames), no clock (font defeats the segment zones), no fouls.
  Validated on 25 minutes of MDN vs Arcadia (scores 0-0 -> 7-18 -> 20, period 1 -> 2). A profile with no `rims`
  section uses the trained rim detector. Each game: perceive ~50 min including the 1 fps scoreboard pass.
  Moon Valley's blue "ROCKETS" skin is a third layout, not fitted. Fitting a skin by hand took ~2 hours; worth
  a tool that finds digit cells from lit columns (the approach that finally worked, see git history).
- Results of the 2026-10-09 batch (all unreviewed, no quarter marks yet):

| Game | Cards | Strong | Safe | From the board | Final score read |
|---|---|---|---|---|---|
| @Cactus Shadows (iPad) | 232 | 88 | 45 | 0 | no board |
| MDN vs St. Mary's | 212 | 84 | 47 | 22 | 58-86 |
| MDN vs Cactus Shadows | 228 | 94 | 47 | 20 | 58-75 |
| MDN vs Arcadia | 405 (about 115 after the final buzzer; the board was not reset) | 140 | 65 | 13 | 40-77 |
| MDN vs Mesquite | 221 | 80 | 36 | 22 | 60-83 |
| MDN vs Saguaro | 226 | 116 | 45 | 27 | 52-? |

- Garrett renamed the 2026-10-08 films (dropped "NEW "); manifests were repointed. Film names are fragile:
  ingest should probably record a content hash, not only a path.
- Runs are chained by a detached script (`run_mdn5.sh` in the session scratchpad; nohup + caffeinate) watched by a
  Monitor, because the session's background-command limit is 2 hours.

## 2026-10-08: four new gyms, run unattended

Garrett downloaded four films to `~/Desktop/Full Games/NEW ...` (new gyms on purpose, "to help you get used to
new gyms") and left, saying do not stop until all are done. Ingested with `--profile handheld` (no scoreboard):

| Game id | Film | Gym and camera (from frames) | `rim@v3` on 8 frames |
|---|---|---|---|
| `2026-10-08-new-arcadia-vs-cortez-angle-1` | 720p, 59 min | Cortez (Austerman Gym), iPad low in the stands; the near backboard is often out of frame | 0 of 8 (0.15-0.23 at low confidence, on the right spot) |
| `2026-10-08-new-marcos-de-niza-vs-tempe-angle-1` | 1080p, 55 min | iPad from the stands, gym not yet identified | 3 of 8 |
| `2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1` | 1080p, 57 min | Cactus Shadows gym (Falcons), iPad from the stands | 4 of 8, right |
| `2026-10-08-new-moon-valley-high-vs-...-tactical` | 1080p, **150 min** | Hudl Focus at Moon Valley, fixed wide view, a different burned-in scoreboard layout (ROCKETS / PERIOD / GUEST) that the Hudl profile does not recognise; backboards small in the top corners; the last ~40 minutes are an empty court | 4 of 8 |

Nobody could label, so `film prelabel rim --auto` was built: the detector's own boxes, kept when found again
0.3 s later, go straight into `datasets/rim`. Then `rim@v4` (30 epochs) and the four games in sequence. He must
still **mark the quarters** on each (Quarters tab) and review; the cards were built without marks.
Results (`rim@v4` after `--auto` on all four gyms: Cortez 42, Tempe 49, Cactus Shadows 90, Moon Valley 68 frames;
0.98 mAP50; 8-frame check went Tempe 3->8, CS 4->5, Moon Valley 4->5, Cortez 0->1):

| Game | Cards | Strong | Safe (>= 0.9) | Disagree | Minutes |
|---|---|---|---|---|---|
| MDN vs Cactus Shadows | 374 | 300 | 184 | 47 | 26 |
| MDN vs Tempe | 128 | 100 | 42 | 35 | 20 |
| Moon Valley vs MDN (150 min film) | 468 (173 typed shot) | 159 | 88 | 52 | 50 |
| Arcadia vs Cortez | 176 | 74 | 53 | 18 | 21 |

Backboard found in 80% (CS), 85% (Tempe), 46% (Moon Valley, wide fixed view) and 48% (Cortez, 720p, near
backboard often out of frame) of frames. All four await Garrett's quarter marks and review.
Moon Valley's last 40 minutes of film (empty court) hold only ~23 cards; quarter marks will clear them.
Game ids carry "new-" and the download date. The Moon Valley overlay is a second Hudl scoreboard layout worth a
profile of its own later (it would give made/missed for free in that gym).

## Fouls on a shot card (built 2026-10-05)

`W` on a shot card cycles **fouled on the shot** / **whistle first** / none; Enter saves it with the decision
(`foul: "on_shot" | "before_shot"` in `review.json`, `FOULS` in `film/review/decisions.py`). The export carries
`foul` and `counts`; `counts` is false for a whistle-first shot, which is not an attempt. The made-or-missed
classifier still trains on what the ball did, so such a card is **confirmed** with its true made/missed, never
rejected. Garrett sends these as the start of foul training data.

Marked so far: Combs `shot-002608-r19`, `shot-033855-r183` (saved rejected), `shot-041248-r205` whistle first;
`shot-008010-r47`, `shot-009368-r50`, `shot-016206-r78`, `shot-029165-r162` fouled on the shot. RKDA
`shot-012883-r85` whistle first. **Not yet marked, kind not stated:** RKDA `shot-014405-r96`, `shot-017324-r124`,
`shot-024920-r172` ("same situation ... in the context of a foul happening"). RKDA was the first game he
flagged; the four earlier games have unflagged ones. Still to build: which team and which player fouled.

## 2026-10-12: shot locations, plan A (court finder and dots on the card)

Plan A of three is built: `docs/superpowers/plans/2026-10-12-shot-locations-a-court-and-dots.md`. Plan B is people,
release and automatic dots; plan C is teams, confidence and eval.

- **Court geometry.** `film/interpret/court.py`: 18 named court points (`near-*`/`far-*`, `POINT_NAMES`), the
  template in feet, `fold` to a half court, `FLIP_INDEX` (left-right mirror pairs, used as `flip_idx`), `Mapping`,
  and `fit_mapping`, a RANSAC homography fitted feet to pixels (so the tolerance is in pixels) and then inverted.
  Needs 4 inliers or returns None. It also returns None for a mirrored labelling (left/right names swapped: the
  Jacobian determinant has the wrong sign), and `to_court` returns None for a pixel at or above the horizon.
  `Mapping.end` is advisory: it is which end the labels called near and is not reliable on a sideline view (fliplr
  augmentation and per-frame labelling); x/y are unaffected because `fold` cancels it, and a shot's end comes from the rim.
- **Zones.** `film/interpret/zones.py`: `zone_of(x, y)` and `shot_type_of(x, y)` on the half court. The arc is the
  NFHS 19.75 ft distance test; a corner three is y < 10 ft.
- **Court finder.** `film/perceive/courtfinder.py`: a YOLO pose model named `court` (`kpt_shape [18, 3]`, imgsz 960).
  `film/perceive/court_run.py` and `film/commands/court.py`: `film court <game_id>` fits the mapping through the
  game and writes `games/<id>/court.parquet`; `mapping_at(mappings, t)` returns the mapping for a time.
- **Labels.** `film prelabel court <game_id>` (`prelabel_court`, 40 frames). The **Label court** tab in the review
  app posts to `/api/label/{ds}/{id}/points`, which calls `accept_points` in `film/train/datasets.py`. Dataset
  layout: `datasets/court`, YOLO pose, 18 points, with `flip_idx` and `kpt_shape` in the data yaml.
- **Training.** `film train court` (base `yolo11n-pose.pt`, imgsz 960, about 30 minutes; `--epochs` to shorten).
- **The card.** A half court on the shot card: click places the dot, arrows nudge it 1 ft, Shift+arrows 5 ft.
  Decisions carry `x`, `y`; events carry `x`, `y`, `zone`, `shot_type`, `location_conf`.
- **Export and eval.** The shots export gains those fields. Eval reports `located`, `dot_placed_by_coach`,
  `median_dot_error_ft` and `zone_agreement`; the last two are null until plan B places machine dots, because
  there is nothing to compare against the coach's dot yet.
- **Not done:** the first real labels, training and a `film court` run on a labelled game (needs Garrett;
  Step 3 of the plan's Task 13). The mapping has not been checked by eye on real film.

## What Garrett has asked for (requirements gathered in conversation)

1. **Full box score eventually**, as automated as possible; he accepts reviewing some of it.
   On 2026-10-09 he widened it: "time of possession, passes, literally anything analytically we can build I
   want to try." Possession and passes need the ball followed across the court and players tracked by team:
   the box-score rung's tracking work unlocks both. Keep a list of such asks here and tick them off.
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
   `JV vs Deer Valley JV - Angle 1.mp4` first (done 2026-10-02), then `Prescott vs Arcadia .mp4`.
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

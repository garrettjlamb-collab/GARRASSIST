# Basketball Film Pipeline — Design

Date: 2026-09-30
Status: design approved section by section in conversation; awaiting review of this written spec
Owner: Garrett Lamb
Companion repo: `~/coaching-sim-web` (the shot tracker and coaching sim), which receives the import and the arc change described in section 9

## 1. Purpose

Turn a downloaded game film into the data the shot tracker and coaching sim consume, with as little hand work as possible. Shots first, then substitutions, lineups and plus-minus, then a full box score. The product model is Hudl Assist: the machine proposes, the coach confirms in a fast review screen. Nothing is silently dropped; anything the machine cannot decide is handed to the coach with the evidence, and at worst with a timestamp into the film.

## 2. Decisions and why

| Decision | Choice | Why |
|---|---|---|
| Film source | Hudl downloads of whatever the gym recorded | It is what exists. Three camera profiles observed, see section 3. |
| Compute | Local only: a Mac mini (M6 or M5 Pro, 24 GB minimum) once purchased, the M3 / 8 GB laptop until then. No rented GPU. | Owner's decision. No uploads, film of minors stays home, overnight turnaround is acceptable. |
| Code portability | Device-agnostic: PyTorch with a device flag (`mps` on Apple silicon, `cuda` if ever present, `cpu` fallback) | A cloud or NVIDIA path can be added later without redesign. |
| Detection approach | Rim-anchored event pipeline (approach 1 of three considered) | Every shot ends at a rim; the rim is large, rigid and easy to detect; a rim crop is a tiny, accurate made-or-missed classifier; the scoreboard double-checks it. It is the only approach that yields measurements (coordinates, tracks, timelines) rather than descriptions, and the only one that grows into a box score. |
| Tracks | Kept in full for every game, not discarded after each shot | The per-frame positions of all players and the ball are the richest layer (spacing, tempo, possession length) and cost tens of MB per game. |
| Player identity | Team plus jersey number is the key for every stat. Names are optional labels. | Opponents have no roster on hand; jersey reading produces numbers; the tracker already adds players number-first. |
| Three-point arc in the tracker | High school (NFHS): 19.75 ft radius, pure arc to the baseline, no straight corner segment | It is what the film shows. Film locations are accurate enough that the arc matters for zone classification. |
| Repos | Pipeline and review page in this repo (Python). Tracker import and arc change in `coaching-sim-web`. | Different language and runtime. |
| Deadline | Ladder rungs 1 and 2 (section 13) working for the first game in late November 2026 | Owner's target. Rung 3 is likely by then; 4 and 5 are in-season work. |
| Training | On the Mac mini, overnight, small models, starting from community weights | No rented GPU. Retraining happens every few games, not daily, so hours are acceptable. |
| Claude pre-labeling | Optional, off by default | It is an API call rather than a rented GPU, saves clicking, and can be skipped entirely. |

## 3. Evidence from the film (observed 2026-09-29)

Files on disk in `~/Downloads` (fully local) and `~/Desktop/Full Games` (iCloud placeholders, not local). Frames were extracted at several timestamps from three files.

| Profile | Example file | What was seen |
|---|---|---|
| `hudl-tactical` | `JV vs Marcos - Tactical.mp4`, 1920x1080, 90 min | Hudl Focus. Digital pan across a fixed panorama: at 25 and 80 minutes the frame shows the left basket, at 40 minutes centre court. Camera never physically moves. Burned-in scoreboard bottom-left with seven-segment digits: home score, clock, visitor score, period, fouls, timeouts, possession arrow. Both rims visible when play is at that end. Bench and crowd in frame. Court lines crisp. Jersey numbers readable near the camera, small far from it. |
| `nfhs-auto` | `JV vs Deer Valley JV - Angle 1.mp4`, 1920x1080, 105 min | NFHS Network auto-camera in the same gym. Pans and zooms to follow play. "NFHS Network" watermark top-right. A boxed overlay bottom-right that was blurred in both sampled frames; no reliable scoreboard assumed. Jersey numbers clearly readable. |
| `nfhs-broadcast` | `State Championship.mp4`, 1280x720, 128 min | Broadcast production with a full score bug: team names, scores, period, game clock, shot clock, fouls. Likely camera cuts. |
| `handheld` | none on disk; the owner reports iPad recordings happen | Assumed: low, wide, panning, no scoreboard. Handled by per-frame geometry and manual quarter marks. |

All observed profiles are H.264 with AAC audio.

## 4. System shape

Five pieces connected by files, each runnable alone and re-runnable.

```
Ingest  ->  Perceive  ->  Interpret  ->  Review & Export  ->  Tracker import (coaching-sim-web)
              ^
            Train  (produces the model weights Perceive uses)
```

Per-game working directory, `games/<game-id>/`:

| File | Written by | Contents |
|---|---|---|
| `manifest.json` | Ingest | source path, duration, resolution, fps, camera profile, frame rate used, model versions used by later stages, stage status |
| `motion.parquet` | Perceive | one record per rim per frame: `frame, t, rim_id, energy, energy_global`, the motion-energy fallback signal |
| `scoreboard.jsonl` | Perceive | one record per second: `t, period, clock_s, home, away, home_fouls, away_fouls, conf` |
| `tracks.parquet` | Perceive | one record per detected object per frame: `frame, t, kind (person/rim/ball), track_id, x1, y1, x2, y2, conf, on_court, court_x, court_y, team, model_version`. `on_court` is false for bench, crowd and table people, who are kept so the substitution rule can see players leave and enter. Court coordinates are the tracker's half-court feet (section 6) and are null where geometry is unavailable or the person is off court. |
| `events.json` | Interpret | list of events (section 7) |
| `review.json` | Review | every decision the coach made: confirmations, fixes, identities, quarter marks, home colour |
| `export/<game-id>.tracker.json` | Export | the file the tracker imports (section 9) |

`<game-id>` is `<date>-<slug of source filename>`. The film stays where it is; the manifest records its path.

The tracks file is the contract between the halves. Everything above it (Interpret, Review, Export) is deterministic and re-runs in seconds. Everything below it (Ingest, Perceive) is the expensive part and runs once per game per model version.

## 5. Ingest

- Input: a path to an `.mp4` or `.mov`.
- Probes duration, resolution and fps with `ffprobe`.
- Samples the first three minutes to choose the camera profile: `hudl-tactical` if the Hudl scoreboard overlay is detected in the bottom-left region; `nfhs-broadcast` if a score bug is detected along the bottom; `nfhs-auto` if the NFHS watermark is detected top-right; otherwise `handheld`. The profile can be overridden on the command line.
- Records the working frame rate: 10 fps by default, 5 fps on the laptop (`--fps`). No frames are written to disk; every later stage decodes on the fly with PyAV at that rate, so a game costs no intermediate storage.
- Writes the manifest. Runs on any machine, no graphics cores needed.

## 6. Perceive

Runtime: Python 3.12, PyAV for decoding, PyTorch on the Metal backend, Ultralytics YOLO for detectors (with YOLO-World open-vocabulary prompts as the fallback before anything is trained), OpenCV for image work. Device chosen by flag; `mps` by default on a Mac. Perceive is a single streaming pass: rims, the ball near each rim, motion energy at each rim, and the scoreboard once a second, all from one decode.

- **Scoreboard reading**, once per second, on the profile's crop box. `hudl-tactical` uses a small seven-segment digit classifier (template or tiny CNN) because the digits are a fixed LED font. `nfhs-broadcast` uses a general OCR model on the bug region. `nfhs-auto` and `handheld` produce no scoreboard timeline. Output is cleaned: within a period the clock must be non-increasing, scores non-decreasing, and single-second outliers are dropped.
- **People and rims, full frame**, every kept frame, at 640 px input. One detector with classes `person`, `rim`, `backboard`. People are tracked with ByteTrack so each gets a stable `track_id`. People whose feet fall outside the court polygon (bench, crowd, table) are kept in the tracks file with `on_court = false`, so the substitution rule (section 7) can see players leaving and entering; every other rule ignores them.
- **Ball, rim-anchored.** For each rim detected in a frame, a native-resolution crop (about 320 px square, centred on the rim) is run through a small ball detector on every kept frame. This is cheap and catches every arrival. When the Interpret stage later identifies a rim arrival, a full-frame ball search at 1280 px runs over the three seconds before it, reading frames already on disk, to recover the arc and the release point. The full-frame search is therefore run only around candidate shots, not over the whole game.
- **Court geometry.** A keypoint model predicts court line intersections (baseline and sideline corners, lane corners, free-throw circle, centre circle, three-point arc ends). A homography from pixels to court feet is fitted with RANSAC. On `hudl-tactical` and `nfhs-*` the camera is fixed and only the crop moves, so the fit is estimated once per second and interpolated between. On `handheld` it is fitted per frame. Where fewer than four confident keypoints are visible, geometry is marked unavailable for that span, and court coordinates are null.
- **Court coordinates.** Both ends of the floor fold onto one half court using the tracker's convention: 50 ft wide, 47 ft deep, basket at `(25, 5.25)`, baseline at `y = 0`, the shooter's own basket. Which end is which is decided by the nearest rim.
- **Team colours.** Torso crops of on-court people are clustered once per game (k-means, k = 3: two teams and referees, referees identified by the stripe cluster). Each track carries the majority cluster of its crops. Clusters are labelled `home` and `away` by the Interpret stage using the scoreboard (section 7), or by the coach in review setup when there is no scoreboard.
- **Output:** `scoreboard.jsonl`, `tracks.parquet`, and the manifest updated with model versions and timings.

Budget: roughly an hour per game on an M5 Pro Mac mini, roughly two on an M6, considerably longer on the M3 laptop at 5 fps. All overnight.

## 7. Interpret

Pure rules over the tracks file and the scoreboard timeline. No graphics cores. Re-run on every game whenever the rules change.

Events are written to `events.json` as a list. Every event has `id, type, t, quarter, clock_s, team, confidence, clip {start_t, end_t}, signals {…}` where `signals` records what each rule saw, so review can show why the machine decided what it did.

- **Shot attempt** (`type: shot`). A ball track that descends into the zone around a rim (a box extending above and slightly below the rim, scaled to the rim's size) after a release moment within the previous four seconds. A release is the frame where the ball separates from the shooter's box and rises. Passes crossing the zone do not descend onto the rim and are not shots. A second arrival with no release between (tip, putback) is its own attempt.
- **Free throw** (`type: free_throw`). Clock stopped, shooter standing within the free-throw circle, at least four other players set along the lane, ball arrival at that rim. Exported as shot type `ft` so the tracker keeps treating it as a non-possession.
- **Made or missed.** Primary signal: the rim-crop classifier on the two seconds around the arrival (`made`, `missed`). Check signal: the shooting team's score rising by 2 or 3 within five seconds of the arrival (or by 1 for a free throw). Agreement gives high confidence. Disagreement, or a 3-point score change on a shot located inside the arc (or the reverse), lowers confidence and flags the event.
- **Shooter and location.** The tracked person nearest the ball at the release frame is the shooter. Their feet (bottom-centre of the box) at release, mapped through the homography and folded to the half court, are the shot location. `zone` and suggested `shotType` are computed with the same rules the tracker uses (ported verbatim, including the high-school arc), so film shots classify exactly as hand-charted shots do. Where geometry is unavailable the location is null and the event is flagged `needs_location`.
- **Home and away.** The colour cluster whose makes coincide with rises in the home score is `home`. Decided from the first few agreeing makes and then fixed for the game. Without a scoreboard, review setup asks once.
- **Quarter and clock** are read from the scoreboard timeline at the event time. Without a scoreboard, quarter comes from the coach's quarter marks and clock is null.
- **Substitution** (`type: substitution`). While the clock is stopped (or, without a scoreboard, while no ball movement toward either rim is seen for at least five seconds), a tracked person leaves the court polygon toward the bench region and another enters from it. The event carries the two `track_id`s, the bench side, and the best thumbnail crops of each (largest, sharpest, most frontal, number region visible).
- **Lineup timeline.** For each team, the set of on-court `track_id`s over time, cut at every substitution and period boundary. Anonymous until review assigns numbers.
- **Track continuity.** When ByteTrack loses a person and starts a new id, Interpret tries to re-link the new track to a recently ended one by position continuity and appearance similarity (torso colour histogram). Unlinked new tracks enter the identity queue like a substitution.
- **Unknown** (`type: unknown`). Anything that looked like an event (a rim arrival with no traceable release, a clock stop with no recognisable cause) but could not be classified. Carries the timestamp and clip so the coach can look.
- **Derived stats**, computed after review assigns numbers: per player minutes, plus-minus, field goal attempts and makes by zone; per lineup minutes, points for, points against, and points per possession using the tracker's possession convention (a possession ends on a field goal attempt, a turnover, or a called set play that produced no shot; free throws do not end one).
- **Confidence.** A number in `[0, 1]` built from how many independent signals agreed (classifier and scoreboard for made; release found and shooter unambiguous for location; both tracks clean for a substitution).

Growth path (not in scope for rungs 1 to 3): rebound as which team's player first holds the ball after a miss; turnover as a possession change with no shot; foul as a clock stop coinciding with a fouls increment on the scoreboard; assist as a pass received within two seconds before a make. These are additional rules over the same tracks file. Player names arrive by adding jersey reading to Perceive; nothing in Interpret changes for that.

## 8. Review

A single-page vanilla JavaScript app (matching the tracker's zero-dependency style) served by a small FastAPI process on the owner's machine, so it can stream the local film and write `review.json`. Started with `film review`. Games are listed; each opens into four queues worked in order, then export.

- **Setup, once per game.** Confirm or override the camera profile. Confirm which jersey colour is home (pre-filled from the scoreboard when available). Set the two tracker team names (pre-filled from the filename; the owner's team name remembered across games). If there is no scoreboard, mark the four quarter starts by seeking the film and pressing a key.
- **Identity queue.** One card per track that needs a number: the starting fives, each player subbing in, and each unlinked track. Ordered by how many events depend on the track. Each card shows three to six best crops with their timestamps. Three tiers:
  1. A jersey reader (when one exists) read the number with high confidence: number pre-filled, `Enter` confirms.
  2. Unsure, or no reader yet: crops shown, the coach types the number.
  3. No usable crop: timestamps shown, one click seeks the film to that moment.
  Skipping leaves the track as `unknown`; its events still count toward the team. A number can be assigned to several tracks (the same player re-tracked). Assigning the same number to two tracks that overlap in time is flagged.
- **Shot queue.** One card per shot and free throw, lowest confidence first. Shows a looping clip of `clip.start_t` to `clip.end_t`, the half court with the proposed dot, made or missed, team, shooter number, and the signals panel. Keys: `Enter` confirm, `M` toggle made, arrows nudge the dot one foot, `T` toggle team, typing digits reassigns the shooter, `L` seeks the film to the shot, `X` rejects (not a shot), `U` marks unsure for later. A "confirm all above 0.9" action exists for the coach who wants to review only the doubtful ones.
- **Substitution queue.** One card per substitution: in and out thumbnails with numbers pre-filled from the identity queue; confirm, swap, or fix.
- **Unknown queue.** The `unknown` events with clips and timestamps; the coach can classify one as a shot (which creates a shot card), a substitution, or dismiss it.
- **Export.** Enabled once every shot card is confirmed, rejected, or marked unsure. Unsure shots export with `needsTags: true` and `unsure: true`. Writes `export/<game-id>.tracker.json` and a derived-stats summary.

`review.json` is the source of truth for coach decisions; Interpret output is never edited in place, so re-running Interpret with better rules re-applies the coach's decisions by event id where the event still exists and queues the rest.

## 9. Export format and tracker changes

### Export file

```json
{
  "format": "film-game-export",
  "schemaVersion": 1,
  "gameId": "2025-02-05-jv-vs-marcos-tactical",
  "game": "JV vs Marcos",
  "date": "2025-02-05",
  "source": { "file": "…/JV vs Marcos - Tactical.mp4", "profile": "hudl-tactical", "modelVersions": { "detector": "…", "rimclf": "…", "court": "…" } },
  "teams": {
    "Arcadia JV": {
      "side": "home",
      "players": ["#1", "#2", "#5"],
      "shots": [ { "id": "film-…", "game": "JV vs Marcos", "date": "2025-02-05", "quarter": 2, "clock": 278, "t": 1500.4, "x": 31.2, "y": 18.7, "zone": "mid-right", "made": true, "shotType": "midrange", "player": "#2", "source": "film", "confidence": 0.96, "needsTags": true, "unsure": false, "clip": { "start": 1498.0, "end": 1502.0 },
                   "actions": [], "action": null, "contest": null, "phases": [], "phaseNote": "", "setPlay": false, "setPlayNote": "", "hand": null } ],
      "substitutions": [ { "t": 1620.0, "quarter": 2, "clock": 210, "in": "#5", "out": "#2" } ],
      "lineups": [ { "startT": 0, "endT": 1620.0, "quarter": 1, "onFloor": ["#1", "#2", "#…"], "pointsFor": 12, "pointsAgainst": 9, "possessions": 14 } ],
      "playerStats": { "#2": { "minutes": 18.5, "plusMinus": 6, "fga": 9, "fgm": 4 } }
    },
    "Marcos JV": { "side": "away", "…": "…" }
  }
}
```

Shot records carry every field the tracker's `saveShot` writes, with the coaching-judgment fields empty, plus `source`, `confidence`, `needsTags`, `unsure`, `clock`, `t` and `clip`, which the tracker stores and ignores except for `needsTags`.

### Tracker changes (in `coaching-sim-web/public/shot-tracker/index.html`)

1. **Import film game.** A button next to Restore. Accepts a `film-game-export` file. For each team: match an existing tracker team by name (case- and whitespace-insensitive) or create it. Match players by leading number against the existing player strings (the tracker stores players as strings such as `23 Name`); create `#23` when missing. Append shots whose ids are not already present, so re-importing a game is idempotent. For each non-free-throw film shot with no possession, create an auto possession the same way `saveShot` does for a hand-charted shot with no active possession, so per-possession rates stay correct. Store `substitutions`, `lineups` and `playerStats` under `team.filmGames[gameId]`. Never replaces existing data.
2. **Needs-tags filter.** The shot log gains a "Needs tags" filter and count. Opening such a shot pre-fills the modal from the record; saving clears `needsTags`.
3. **High-school arc.** `COURT.threeRadius` becomes `19.75`. `isThreePointer` becomes a plain distance test (the NFHS arc meets the baseline; there is no straight corner segment), `cornerThreeX` is removed, and `drawCourt` draws the arc to the baseline. `zone` is a derived field, so on first load after this change every stored shot's `zone` is recomputed from its `x, y` under a bumped schema version; `shotType` is the coach's choice and is left alone. Backups made before the change import cleanly because the recompute runs on load.
4. **Lineup and plus-minus views** consume `team.filmGames` and are specified separately once rung 3 lands; they are out of scope for this spec beyond the storage location.

## 10. Training and labeling

- **Starting weights.** `person` from public COCO-trained YOLO weights as is. Ball, rim/backboard and court keypoints start from community basketball datasets and weights (Roboflow Universe and similar), then adapt on the owner's film. The rim-crop made-or-missed classifier starts from nothing.
- **Label sources.** The review page is the main labeling tool: every confirmed or fixed shot is a made-or-missed label and a shot-or-not label; every typed number is a jersey label for a future reader; every confirmed substitution is a sub label. For ball and rim boxes the review app has a box-drawing mode; the pipeline picks the frames to label (active learning: the lowest-confidence detections and a random sample), so labeling stays at a few hundred frames per camera profile.
- **Optional Claude pre-labeling.** A `film prelabel` command can send selected frames or rim clips to Claude with a fixed prompt and write proposed labels for the coach to confirm. Off unless invoked.
- **Training on the Mac mini.** `film train <detector|rimclf|court|scoreboard>` fine-tunes on the Metal backend: nano or small YOLO variants at 640 px, of the order of an hour or two per detector; minutes for the rim classifier. Intended to run overnight after a few games' corrections.
- **Versioning.** `models/<name>/<version>/` holds weights plus `manifest.json` recording training data (game ids and label counts), hyperparameters, metrics and date. The game manifest and tracks file record which versions produced them, so a game can be re-perceived when models improve.
- **Metrics**, against hand-charted games exported from the tracker as the test set: shot recall, shot precision, made-or-missed accuracy, median location error in feet, team accuracy, and for substitutions, recall and precision against the coach's review. Targets for calling shots-only "ready": recall and precision at or above 0.95, made-or-missed at or above 0.97, median location error at or below 3 ft, team at or above 0.99. These are targets, not guarantees; the report is printed by `film eval <game>`.

## 11. Failure handling

- Every stage writes its output and updates its status in the manifest; a stage can be re-run alone with `--force`.
- A wrong camera profile is overridden on the command line or in review setup, and Ingest re-runs.
- No scoreboard or an unreadable one: quarter comes from review quarter marks, clock is null, home is asked in setup, made-or-missed relies on the classifier alone with lower confidence.
- Missing geometry: shots carry a null location and `needs_location`; the shot card asks the coach to place the dot.
- Lost tracks: re-link where possible, otherwise the identity queue.
- Anything unclassifiable: an `unknown` event with a timestamp, never a silent drop.
- No frames are stored; anything that needs pixels again (montages for the classifier, clips for review, frames for labeling) seeks into the film and decodes the few seconds it needs.
- The laptop path: same code, `--fps 5`, nano models, and a warning printed with the expected runtime.

## 12. Testing

- **Interpret rules:** unit tests on hand-built tracks and scoreboard timelines with no video, covering shot detection, pass rejection, tip-ins, free throws, made-or-missed agreement and disagreement, home-and-away inference, substitution detection, lineup cutting, track re-linking, and derived stats. Run in seconds.
- **Perceive:** fixture clips of about thirty seconds cut from the owner's film by a script (the clips themselves are not committed; the script and expected outputs are). Each fixture has an expected scoreboard read, rim count, and ball arrivals.
- **End to end:** one full hand-charted game compared with the export using `film eval`. The owner exports the tracker backup for a game that exists on disk to provide the reference.
- **Tracker:** the coaching-sim repo's existing Vitest suite gains tests for the import (merge, idempotence, player matching, auto possessions), the needs-tags filter, and the arc change including zone recompute.

## 13. The ladder

Each rung is useful on its own and shippable alone.

1. **Shot times.** Ingest, scoreboard reading, rim detection, rim-crop ball arrival, made-or-missed classifier, on `hudl-tactical` film. Output: every shot's time, result and quarter, and a clip reel. Review shot queue without locations.
2. **Shots into the tracker.** People, court geometry, shooter location, team colours, the full shot card, export, and the tracker import with the needs-tags filter and the high-school arc. This is "shots only".
3. **Lineups.** Substitution events, the identity queue by number, lineup timeline, minutes and plus-minus, lineup points for and against.
4. **More cameras and a jersey reader.** `nfhs-auto`, `nfhs-broadcast` and `handheld` profiles; a first jersey-number reader that pre-fills identity cards.
5. **Box score.** Rebounds, turnovers, fouls, assists.

Rungs 1 and 2 are the late-November commitment. Rung 3 is likely by then. Rungs 4 and 5 are in-season work.

Each rung gets its own implementation plan under `docs/superpowers/plans/`, written from this spec when that rung starts. The first plan, `2026-09-30-rung-1-shot-times.md`, covers rung 1; rung 2 and the tracker changes of section 9 follow as their own plans.

## 14. Stack and repo layout

Python 3.12 managed with `uv`; PyAV for decoding and probing and the ffmpeg binary bundled with `imageio-ffmpeg` for cutting clips (no Homebrew needed); PyTorch (Metal backend); `ultralytics`; `opencv-python`; `numpy`; `pandas` and `pyarrow`; `fastapi` and `uvicorn` for the review server; `pytest`. A single CLI, `film`, with subcommands `ingest`, `perceive`, `interpret`, `review`, `export`, `eval`, `train`, `prelabel`.

```
basketball-film-pipeline/
  pyproject.toml
  film/
    cli.py
    commands/      one file per subcommand
    ingest/        probe, profile detection
    perceive/      scoreboard, detector, tracker, ball, court, colours
    interpret/     rules, events, lineups, stats
    review/        FastAPI app + static single-page review UI
    export/        tracker export
    train/         fine-tuning and eval
    common/        manifest, paths, device selection, video decoding, model registry, court model (ported from the tracker)
  models/          versioned weights (git-ignored except manifests)
  games/           per-game working directories (git-ignored)
  fixtures/        fixture scripts and expected outputs (clips git-ignored)
  tests/
  docs/superpowers/specs/
```

## 15. Non-goals for this spec

Real-time or in-game use; player names as a requirement; plays, coverages or defensive assignments; a mobile app; multi-device sync; a cloud path (kept possible, not built); replacing the coach's contest, action and phase tags.

## 16. To confirm during implementation (not blockers)

- Which past games are hand-charted in the tracker, to serve as the evaluation set. Candidates on disk: `JV vs Marcos - Tactical`, `JV vs Deer Valley JV - Angle 1`, `Arcadia Semi Final`, `State Championship`.
- The owner's team name as it appears in the tracker, for the setup default.
- Whether Claude pre-labeling will be used at all.

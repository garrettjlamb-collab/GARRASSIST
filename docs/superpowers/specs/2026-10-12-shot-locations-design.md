# Shot locations, teams and the export: design

Written 2026-10-12 with Garrett. Rung 2 of the pipeline (see `2026-09-30-film-pipeline-design.md`, section 13):
every shot already found gets a spot on the tracker's half court, a zone, and a team, reviewed on the card and
written to the export. Home gym this season is Marcos de Niza (Hudl Focus, `hudl-mdn` skin); away games are iPad
or NFHS film in other gyms. Both teams' shots are charted. No hand-charted answer key exists: the review page's
clicks are the answer key and the training data.

## 1. What gets built

Four stages, each re-runnable on its own like the existing ones.

1. **Court finder** (`film/perceive/court.py`, model `court`): court line intersections per frame, a
   pixels-to-feet mapping, a confidence.
2. **People and teams** (`film/perceive/people.py`, `film/interpret/teams.py`): tracked people, on-court or not,
   grouped by jersey colour into two teams and referees.
3. **Shooter and spot** (`film/interpret/release.py`, `film/interpret/location.py`): the release frame from the
   ball's arc traced back from the rim, the shooter, the spot in feet, the zone and shot type, a confidence.
4. **Review and export**: a half court on the shot card with the dot and team, keys to nudge or place; the shots
   export carries x, y, zone, shot type, team and confidences.

Build order, so something usable lands early: court finder and the half court on the card (dots can be placed
by hand from day one) -> shooter and spot -> teams -> export. Each step is measured against the coach's clicks
before the next starts.

## 2. Court finder

**Points.** Up to 14 per frame: the four court corners; at each end in view, the four lane corners, the two
free-throw line ends, the top of the free-throw circle, and the two points where the three-point arc meets the
baseline. (The NFHS arc, 19.75 ft, meets the baseline; there is no straight corner segment.)

**Model.** A keypoint model in the same family as the backboard detector (YOLO pose/keypoint, nano), trained
on frames labelled in a new **Label court** tab: the frame is shown with the 14 points listed; the coach clicks
each visible one and skips the rest. About 40 frames per gym, spread across the game so every camera position
is covered; `film prelabel court <game_id>` queues them, `film train court` trains. Labels are stored like the
box datasets (`datasets/court`, YOLO keypoint layout). As with backboards, a gym the model has seen needs no
labels; a new gym may need fewer than 40 once several gyms are in.

**Mapping.** From four or more confident points, a homography from pixels to court feet, fitted with RANSAC
against the court template. Both ends fold onto the tracker's half court: 50 ft wide, 47 ft deep, basket at
(25, 5.25), baseline at y = 0; the end is chosen by the rim the shot was at. On the Hudl profiles (digital pan
over a fixed panorama) the mapping is fitted once a second and interpolated; on `handheld` and `nfhs-*` it is
fitted on every frame the model is confident on and carried forward through gaps of up to 2 s.

**Confidence.** From the number of points found and the RANSAC reprojection error. Below a threshold the
stretch has no geometry (`geometry` absent in `tracks.parquet` for those frames) and shots there are flagged
`needs_location`.

**Measured** on held-out labelled frames: pixel error of predicted points, and the floor error of a known spot
(the free-throw line's midpoint) through the fitted mapping. Target: median floor error under 3 ft on the Hudl
cameras; the iPad figure is reported.

**Output.** `games/<id>/court.parquet`: per fitted frame, the homography (9 numbers), the end in view, point
count, error, confidence.

## 3. People and teams

**People.** A standard COCO person detector (YOLO nano, no training) on every kept frame at 640 px, tracked
with ByteTrack so each person keeps a `track_id`. Rows go into `tracks.parquet` with `kind = "person"`.
`on_court` is true when the feet, through the court mapping, land inside the court polygon (with a 2 ft
margin); bench, table and crowd are kept with `on_court = false` for the later substitution work and ignored by
everything in this spec. Without geometry, `on_court` is null and a person counts as on court for shooter
selection.

**Teams.** Once per game, torso crops (the middle third of each on-court person box) are sampled across the
game and clustered by colour (k-means, k = 3, in a colour space that ignores brightness). The referee cluster
is the one with the highest stripe response (alternating dark/light columns). Each track carries the majority
cluster of its crops as `team` (`a`, `b` or `ref`). **Setup question**, once per game on the review page: a
crop from each of the two team clusters, "which is <home team name>?" (names pre-filled from the game title);
the answer maps `a`/`b` to `home`/`away` in `games/<id>/setup.json`. On a game with a scoreboard, the side that
scored after a confirmed make is compared with the shooter's cluster as a sanity check and logged. Without the
setup answer, cards show the cluster letter and the export says `team: null`.

## 4. Shooter, release and the dot

**Release.** For each shot card (ball or motion arrival, and scoreboard-only cards placed at their estimated
time), the ball detector runs on the full frame at 1280 px for the 2.0 s before the arrival, in addition to
the rim crop it already runs on. Detections are chained backwards from the arrival into an arc (each step
within a plausible distance of the last, moving away from the rim going backwards). The earliest arc point
within 1.5 person-box-widths of a tracked person is the release; its frame is the release frame.
**Fallback:** no arc -> release at arrival minus 1.2 s, flagged `release-estimated`.

**Shooter.** At the release frame, the tracked on-court person whose box top-centre is nearest the ball;
on the fallback, the on-court person nearest the line from the arrival point back along the ball's last seen
direction. Ties or two people within a box-width: the nearer one, flagged `shooter-ambiguous`.

**Spot.** The shooter's feet (bottom-centre of the box) at the release frame through the court mapping at
that frame, folded to the half court. `zone` and suggested `shot_type` come from the tracker's rules, ported
verbatim with the high-school arc (`film/interpret/zones.py`; the port is covered by tests that compare it
with the tracker's JavaScript on a grid of points). **Free throws:** clock stopped (or no ball movement for
3 s on clockless film) and the shooter within 2 ft of the free-throw line -> placed at the line's midpoint,
`shot_type = "free throw"`.

**Confidence** of the location, in [0, 1]: geometry confidence x (1.0 arc found / 0.6 fallback) x (1.0 one
clear shooter / 0.7 ambiguous). A dot within 1.5 ft of a zone boundary carries the flag `zone-boundary`.

**Ball mid-court.** The ball detector has only seen rim crops. `film prelabel ball --release` queues full-frame
crops around estimated release moments; a first batch of about 150 is expected, after checking whether the
current detector already copes.

**Events.** Each shot event gains `x`, `y`, `zone`, `shot_type`, `release_t`, `shooter_track`, `team`,
`location_conf`, and the flags above. The `needs_location` flag already exists.

## 5. Review card and export

**Card.** A half-court drawing beside the clip with the dot, the zone name and the team colour. Keys: arrow keys
nudge the dot one foot; a click on the drawing places it; **T** flips the team (now `home`/`away`/blank);
existing keys unchanged. Decisions gain `x`, `y`, `team` as set by the coach (`review.json`). Cards with
`needs_location`, `release-estimated` or `zone-boundary` show why. **Queue order:** undecided strong cards first
as now, with cards that need a dot ahead of those that have one. **Confirm all above 0.9** keeps working: a
card's confidence is now the minimum of made/missed confidence, location confidence and team confidence, so a
0.9 card is confident on all three.

**Export.** `film export` writes, per shot, `x`, `y` (feet, tracker convention), `zone`, `shot_type`, `team`
(`home`/`away`/null), `shooter` (null until jersey numbers exist), `location_conf`, `team_conf`, alongside the
existing fields. The tracker import is the next spec; this one ends at the file.

## 6. Measuring and testing

`film eval <game>` prints, against the coach's clicks on games the models did not train on: median dot error
in feet, share of zones right, share of teams right, and the share of cards that needed a dot placed by hand.
Targets from the original spec: 3 ft median, team 0.99.

Tests: unit tests on hand-built tracks, mappings and arcs with no video (release chaining, shooter choice,
folding both ends, zone port against the tracker's rules, team mapping, confidence rules, queue order); a few
fixture frames for the court finder and the homography fit; the review API for dot and team decisions; the
export format. The coach's reviews are the end-to-end check, as with made-or-missed.

## 7. What this does not do

Jersey numbers and the shooter's identity; substitutions, lineups and plus-minus; rebounds, turnovers, steals,
blocks, possessions, passes; the tracker-side import. Each follows in its own spec.

## 8. Risks

- The ball mid-court on the wide Hudl view is small; many Hudl shots may take the fallback. The fallback's
  accuracy is the first thing `film eval` will show.
- Court labelling: 40 frames per gym for nine gyms is about an hour of Garrett's time in total.
- Colour clustering fails on teams in similar colours; the setup question and **T** cover it, and the
  sanity check against the scoreboard logs how often.

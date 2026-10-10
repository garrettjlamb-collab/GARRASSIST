# Basketball film pipeline

Turns a downloaded game film into shot data for the shot tracker: every shot's time, made or missed,
quarter and clock, confirmed on a local review page, exported as a shot list and a clip reel. The coach's
confirmations become the training data that makes the next game better.

- **New here? Read `docs/HANDOFF.md` first**: current state, measured results, what is not built, next steps.
  The work in progress is the no-scoreboard path (NFHS camera and iPad film); its "Start here" section says
  where to begin.
- Design: `docs/superpowers/specs/2026-09-30-film-pipeline-design.md`.
- Plan and execution ledger for what exists: `docs/superpowers/plans/`.

## What it handles today

| Film | Supported |
|---|---|
| Hudl Focus "Tactical" export from the Arcadia gym (burned-in scoreboard, bottom left) | Yes |
| Anything without that overlay: NFHS Network camera, iPad recordings | Not yet; `film run` stops with a message instead of spending half an hour for nothing |

## Setup (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # no Homebrew needed
export PATH="$HOME/.local/bin:$PATH"                # every new terminal
uv sync --extra dev
uv run film doctor                                   # expect torch ... mps=True on a Mac
```

## A game, end to end

```bash
uv run film run ~/Downloads/"JV vs Marcos - Tactical.mp4" --date 2025-02-05
uv run film review                                   # http://127.0.0.1:8765
uv run film export 2025-02-05-jv-vs-marcos-tactical --reel
uv run film eval 2025-02-05-jv-vs-marcos-tactical
```

`film run` is `ingest` + `perceive` + `interpret` + `clips`. A 90 to 100 minute game takes about half an
hour on the 8 GB laptop, almost all of it in `perceive`. Finished stages are skipped on a re-run; `--force`
redoes them. `--limit-s 120` is a two-minute smoke test (a later run without it redoes the whole game).
`interpret` and `clips` re-run in seconds to minutes (the first `interpret` with a made-or-missed classifier in
service reads each shot's ball path from the film, about 2 minutes) and are safe to repeat: review decisions follow each
shot to its re-detected card and are never deleted.

Nothing here uses the internet or any paid service once the weights are downloaded, with one exception:
a film kept in iCloud only (most of `~/Desktop/Full Games`) is pulled down when a command needs it and put
back when the command ends. It says so when it does, and refuses if the disk has no room. A film you keep
on the Mac is never offloaded. `uv run film doctor` lists where each game's film is.

`uv run film title <game_id> "MDN vs St. Mary's (Hudl)"` names a game for the review page; the id stays.

## Reviewing

`uv run film review` serves the page; add `#shots`, `#rim`, `#ball` or `#rimclf` to the address to open a
tab directly. On the **Shots** tab each card loops an 11-second clip (8 s before the shot, 3 s after) that
opens 3 s before the shot.

| Key | Does |
|---|---|
| Enter | Confirm the card as shown |
| M | Flip made / missed |
| T | Cycle team: home, away, blank |
| W | Foul on the play: fouled on the shot, whistle first (not an attempt), none |
| X | Reject: not a shot |
| U | Unsure |
| K / J | Next / previous without deciding (use K for a warmup clip) |
| L | Open the full film at the start of the clip |
| R | Replay the whole play from the start |
| F | Playback speed: 1x, 1.5x, 2x |

Cards come in game order: ball and scoreboard cards first, the weaker motion-only cards after, warmup
cards last. Warmup time (pregame, breaks, halftime, post-game) is detected from the scoreboard clock;
warmup cards never count in the export or in `film eval`.

## Making it better: label, then train

The detectors start from fallbacks and improve with labels. On Hudl film the rim needs no training (it is
found by matching templates of this gym's backboards), so the labels that matter are the ball and the
made-or-missed clips.

1. **Review a game first.** Your decisions drive everything below.
2. **Ball.** `uv run film prelabel ball <game_id>` queues rim crops from the moments around the shots you
   confirmed; in **Label ball** drag a box round the ball and press S, or K when there is none (a skipped
   crop is never shown again). Then `uv run film train ball`: about 15 minutes on the laptop.
   `uv run film prelabel ball <game_id> --after-shots` instead queues crops from the second after each confirmed
   shot where the detector lost the ball, which is what the made-or-missed classifier depends on. Those come
   with no proposed box: box the ball and press S, or press S with no box when there is no ball in the crop
   (that teaches the detector too); K only when you cannot tell.
3. **Made or missed.** `uv run film prelabel rimclf <game_id>` (about 4 minutes) files a clip for every decided
   card straight into made, missed or none, stores the ball's path after the rim next to it, and queues the
   rest for **Label clips** (1 / 2 / 3). Then `uv run film train rimclf` (seconds). The classifier reads the
   ball's path, not the picture: a make drops straight down under the hoop, a miss goes sideways or back up.
   With two or more games it is scored by training on the others and testing on the one held out; a
   classifier that does not clearly beat guessing is kept out of service and the scoreboard stays in charge.
   `uv run film train picclf` (about 17 minutes) trains the second opinion, a classifier that looks at sixteen
   frames of the hoop and net. It needs two reviewed games. Where the two agree strongly on a card with no
   scoreboard, the card's confidence is 0.92 and **Confirm all above 0.9** accepts it; where they disagree the
   card is flagged `classifiers-disagree`.
4. **Rim, and not on Hudl film.** For other cameras: `uv run film prelabel rim <game_id>`, the **Label rim**
   tab, about 150 frames, `uv run film train rim`. A trained rim model outranks the templates once it
   exists; set `FILM_RIM_DETECTOR=template` to keep the templates on Hudl film.

New versions land in `models/<name>/vN/` and are picked up automatically. Re-run `film perceive` (half an
hour) and `film interpret` on a game to use a new detector; `film interpret` alone for a new classifier.

## Where things live

| Path | What |
|---|---|
| `games/<game_id>/` | manifest, tracks, motion, scoreboard, events, clips, `review.json` (the coach's decisions), exports |
| `labels/<dataset>/pending/` | what the Label tabs show next; `skipped.txt` remembers skips |
| `datasets/<dataset>/` | accepted labels in YOLO or ImageFolder layout |
| `models/<name>/vN/` | trained weights and their manifest; `models/base/` holds downloaded base weights |
| `fixtures/frames/` | real frames used by the tests |

All four data roots are git-ignored. Set `FILM_GAMES_DIR`, `FILM_MODELS_DIR`, `FILM_DATASETS_DIR`,
`FILM_LABELS_DIR` to move any of them, for example onto an external drive.

## Tests

```bash
uv run pytest
```
No test needs weights, the internet, or a GPU.

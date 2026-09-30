# Basketball film pipeline

Turns a downloaded game film into shot data for the shot tracker. Rung 1 (this repo today): every shot's
time, made or missed, quarter and clock, reviewed on a local page, exported as a shot list and a clip reel.
The design is in `docs/superpowers/specs/2026-09-30-film-pipeline-design.md`.

## Setup (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # no Homebrew needed
uv sync --extra dev
uv run film doctor                                   # expect torch ... mps=True on a Mac
```

## A game, end to end

```bash
uv run film run ~/Downloads/"JV vs Marcos - Tactical.mp4" --date 2025-02-05
uv run film review                                   # opens http://127.0.0.1:8765
uv run film export 2025-02-05-jv-vs-marcos-tactical --reel
uv run film eval 2025-02-05-jv-vs-marcos-tactical
```

`film run` is `ingest` + `perceive` + `interpret` + `clips`. Each can be re-run alone. Add `--limit-s 120`
for a two-minute smoke test, `--fps 5` on the 8 GB laptop. A full 90-minute game takes about half an hour
on the laptop.

## Making it better: label, then train

Until you train, the pipeline runs on fallbacks (a template matcher for the rim, open-vocabulary
detection for the ball) and the scoreboard alone. On Hudl film the rim detector already works without
training, because it matches templates of this gym's backboards, so the ball and made-or-missed labels are
the ones that matter most. Each of these makes it faster and more accurate:

1. `uv run film prelabel rim <game_id>` then the **Label rim** tab, on two or three games, at least 150 frames. Then `uv run film train rim` (overnight).
2. `uv run film prelabel ball <game_id>` on a game with events, then **Label ball**, at least 300 crops. Then `uv run film train ball`.
3. `uv run film prelabel rimclf <game_id>` then **Label clips** (made / missed / not a shot). Then `uv run film train rimclf`.

New versions land in `models/<name>/vN/` and are picked up automatically. Re-run `film perceive` and
`film interpret` on a game to use them.

## Where things live

| Path | What |
|---|---|
| `games/<game_id>/` | manifest, tracks, motion, scoreboard, events, clips, review decisions, exports |
| `labels/<dataset>/pending/` | what the Label tabs show next |
| `datasets/<dataset>/` | accepted labels in YOLO or ImageFolder layout |
| `models/<name>/vN/` | trained weights and their manifest |
| `fixtures/frames/` | real frames used by the tests |

Set `FILM_GAMES_DIR`, `FILM_MODELS_DIR`, `FILM_DATASETS_DIR`, `FILM_LABELS_DIR` to move any of these,
for example onto an external drive.

## Tests

```bash
uv run pytest
```
No test needs weights, the internet, or a GPU.

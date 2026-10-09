# Cloud-only film: pull on demand, put back when done — Design

Date: 2026-10-09
Status: design approved in conversation; awaiting review of this written spec
Owner: Garrett Lamb
Branch: `cloud-film` (worktree at `.superpowers/worktrees/cloud-film`)

## 1. Purpose

Game films live in `~/Desktop/Full Games`, which syncs to iCloud Drive. The laptop has a 228 GB disk and each film is 0.6 to 5.7 GB, so most films have to sit in iCloud with no local copy ("cloud-only"). Every processing stage reads the original film, so today a cloud-only film is silently re-downloaded by macOS the moment a stage opens it, with no warning, no space check, and nothing ever puts it back.

This change makes that deliberate: a pipeline job pulls its film down when it needs it, says so, and returns the film to cloud-only when the job ends.

## 2. Decisions and why

| Decision | Choice | Why |
|---|---|---|
| When a film goes back | When the job that pulled it ends | Garrett's choice (2026-10-09). Lowest disk use. Review does not need the film: cards play pre-cut clips from `games/<id>/clips`. |
| Which films go back | Only films the pipeline pulled down | A film Garrett keeps on the Mac must stay on the Mac. |
| Where the logic sits | One module every command goes through, held for the whole command | Wrapping each read (`iter_frames`, `first_frame_at`, `cut_clip`) would download and offload the same film several times inside one `film run`. |
| Full film in review | Pulled on first use, put back after 30 minutes without full-film viewing or when the server stops | The review server runs for long stretches, so "when the server stops" alone would hold films for days. |
| Mechanism | `brctl download` and `brctl evict`, cloud-only detected by the `SF_DATALESS` file flag | Tested on this Mac (macOS 26.3.1) on 2026-10-09: a 16 MB file round-tripped byte-identical, download took 3 s. |

## 3. What was observed on 2026-10-09

- A cloud-only file has `st_flags & 0x40000000` (`SF_DATALESS`) set and `st_blocks == 0`. Python 3.12's `stat` module has no name for the flag, so the module defines the constant.
- `brctl evict` on a file that has not finished uploading fails with `NSFileProviderErrorDomain Code=-2008 "cannot be evicted"`. This is the safety net: a film cannot be offloaded before iCloud holds it.
- `brctl evict` issued immediately after a download returned success and the file stayed local. Success from the command is not proof; the flag must be checked afterwards.
- Files under a dot-prefixed folder are never uploaded, so they can never be evicted.
- A film with "Keep Downloaded" set is always local, so the pipeline never pulls it and therefore never puts it back.

## 4. The module: `film/common/cloud.py`

```python
def is_cloud_only(path: Path) -> bool
@contextmanager
def local_source(source: Path, log=print) -> Iterator[Path]
def film_state(source: Path) -> str          # "local" | "cloud" | "pulled" | "missing"
def hold(source: Path, log=print) -> Hold    # for the review server; Hold.release()
```

`local_source` is the only thing commands use. It yields the same path it was given.

**On entry**
1. Take a hold on the film (section 5).
2. If the film is cloud-only: check free space, log `pulling <name> (<size>) from iCloud`, run `brctl download`, and poll the flag every 2 s, logging elapsed time every 30 s. Record that the pipeline pulled it.
3. If the film is already local, or is not an iCloud file at all (Downloads, an external drive), do nothing.

**On exit** (normal return, exception, or Ctrl+C)
1. Release the hold.
2. If no live hold remains and the film is recorded as pulled: run `brctl evict`, poll the flag for up to 30 s, retry once, then clear the record.
3. If the film is still local after the retry, log one line, `<name> is still on the Mac: macOS would not offload it`, and keep the record so the next job that touches the film tries again.

**Failures, all raised as `CloudError` before any stage work starts**
- Free space under the film's size plus a 5 GB cushion (`FILM_CLOUD_CUSHION_GB`).
- `brctl download` exits non-zero (no network, signed out of iCloud).
- The film is not local after 60 minutes (`FILM_CLOUD_TIMEOUT_S`).

Commands catch `CloudError` at the top, print its message to stderr, and return exit code 2.

The two `brctl` calls sit behind a small `Backend` object passed in or taken from a module default, so tests replace it.

## 5. Holds: two jobs, one film

State is kept per film, not per game, because ingest needs the film before a game id exists:

```
games/.cloud/<sha1 of the resolved film path>/
    lock            # flock'd around every state change
    pulled          # exists when the pipeline pulled this film down
    holds/<pid>-<n> # one file per active local_source
```

- A second `local_source` on the same film, in the same process or another one, adds its own hold. This is what lets `film run` hold the film across ingest, perceive, interpret and clips while each stage also wraps itself.
- A hold whose pid is no longer running is deleted the next time anything touches that film. A job killed with `kill -9` therefore leaves the film local with `pulled` set, and the next job on that film puts it back at its end.
- The flock covers the decision and the `brctl evict` call, so two jobs ending together cannot both decide the other still holds it.

## 6. Where it is wired in

| Command | Function that takes the hold |
|---|---|
| `film run` | `commands/run.py: run`, around the whole body |
| `film ingest` | `ingest/run.py: ingest` |
| `film perceive` | `perceive/run.py: perceive` |
| `film interpret` | `interpret/run.py: interpret` |
| `film scoreboard` | `commands/scoreboard.py: run` |
| `film prelabel` | each public function in `train/prelabel.py` that reads frames |
| `film train` (rimclf dataset) | `train/rimclf_dataset.py`, one game at a time: a film is put back before the next is pulled |
| `film clips` | `export/clips.py: clips_for_game`, and skipped entirely when no clip needs cutting |

## 7. Review page

- Cards are unchanged; they play `games/<id>/clips/*.mp4`.
- New `GET /api/games/{id}/film` returns `{"state": "local" | "pulling" | "cloud", "size_bytes": n}`.
- `GET /video/{id}` on a cloud-only film starts the pull on a background thread and returns 503 with a plain message. The pull uses `hold`, so the server holds the film like any other job.
- The full-film view and the L key call `/api/games/{id}/film` first. While the film is not local the page shows `Pulling the full film from iCloud (<size>)…`, polls every 5 s, and loads the video when it is ready.
- Every `/video/{id}` request stamps that film's last-use time. A timer releases a film's hold 30 minutes after its last request (`FILM_CLOUD_IDLE_S`), and server shutdown releases every hold.

## 8. `film doctor`

Adds one block listing every game's film as `local`, `cloud`, `pulled` (on the Mac because the pipeline pulled it and has not put it back), or `missing`.

## 9. Testing

- Unit tests against a fake backend and a patched `is_cloud_only`, one per rule: pulls only when cloud-only; never offloads a film that started local; non-iCloud paths untouched; nested holds offload once; a second process's hold blocks the offload; a dead pid's hold is ignored; too little space raises before download; download failure and timeout raise; an offload that does not take is logged and retried by the next job; exit paths on exception and KeyboardInterrupt.
- Review app tests for the three film states, the 503, and the idle release with a short timeout.
- `film run` test: one pull and one offload across all four stages.
- One real round trip on `Arcadia vs Cortez - Angle 1.mp4` (0.6 GB), run by hand with Garrett's go-ahead, not in the suite.

## 10. Docs

`docs/HANDOFF.md` gains a short section: films may be cloud-only, any script that reads a film must go through `local_source`, and never open `m.source` directly. The README's film section gets two lines saying the same.

## 11. Out of scope

- Moving films from `~/Downloads` into iCloud, or changing any manifest's `source` path.
- Offloading films that were already on the Mac, or a manual offload command.
- A download progress percentage (macOS does not expose one to `brctl`; elapsed time is logged instead).

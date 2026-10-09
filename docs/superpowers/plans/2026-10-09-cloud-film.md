# Cloud-only Film Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pipeline job pulls a cloud-only film down from iCloud when it needs it and returns it to cloud-only when the job ends.

**Architecture:** One module, `film/common/cloud.py`, owns detection, download, offload and cross-process holds. Stages take a hold through a context manager (`local_source`, or `lazy_source` where the film is only sometimes read). The review server keeps its own holds in `film/review/films.py` and releases them after 30 idle minutes.

**Tech Stack:** Python 3.12, `brctl` (macOS), `fcntl.flock`, FastAPI, pytest.

Spec: `docs/superpowers/specs/2026-10-09-cloud-film-design.md`

## Global Constraints

- Only a film the pipeline pulled is ever offloaded. A film that was local when the job started is never touched.
- A path that is not cloud-only (Downloads, an external drive, a test tmp dir) costs one `os.stat` and nothing else.
- `brctl evict` returning success is not proof. Check the `SF_DATALESS` flag (`0x40000000`) afterwards.
- No test calls `brctl`. Tests replace `cloud.BACKEND` and `cloud.is_cloud_only`.
- Env knobs: `FILM_CLOUD_CUSHION_GB` (default 5), `FILM_CLOUD_TIMEOUT_S` (default 3600), `FILM_CLOUD_IDLE_S` (default 1800).
- Run tests from the worktree with the main checkout's interpreter: `../../../.venv/bin/python -m pytest -q`.
- Commit messages are plain sentences and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not run `brctl` against any file in `~/Desktop/Full Games` without Garrett's go-ahead.

---

### Task 1: `film/common/cloud.py`

**Files:**
- Create: `film/common/cloud.py`
- Test: `tests/test_cloud.py`

**Produces:**
```python
SF_DATALESS = 0x40000000
class CloudError(RuntimeError)
class Backend:                       # brctl; replaced in tests via cloud.BACKEND
    def download(self, path: Path) -> None      # raises CloudError on non-zero exit
    def evict(self, path: Path) -> bool         # True when brctl exits 0
BACKEND = Backend()
def is_cloud_only(path: Path) -> bool
def film_state(source: Path) -> str             # "missing" | "cloud" | "pulled" | "local"
def hold(source: Path, log=print) -> Hold       # Hold.release() is idempotent
@contextmanager
def local_source(source: Path, log=print) -> Iterator[Path]
@contextmanager
def lazy_source(source: Path, log=print) -> Iterator[Callable[[], Path]]   # holds on first call
```

State lives in `games_dir() / ".cloud" / sha1(str(resolved source))[:16]` as `lock`, `pulled`, `holds/<pid>-<8 hex>`.

`hold`: under `flock(lock)`: drop holds whose pid is dead, write this hold, then if `is_cloud_only(source)`: check `shutil.disk_usage(source.parent).free >= size + cushion`, log `pulling <name> (<n.n> GB) from iCloud`, `BACKEND.download`, poll every `POLL_S` (2 s) until not cloud-only, log `  still pulling <name>: <m> min` every 30 s, raise after the timeout, then touch `pulled`. Any failure removes the hold file before raising `CloudError`.

`Hold.release`: under the lock: remove this hold, drop dead holds; if none remain and `pulled` exists: if the film is already cloud-only clear `pulled`; else `BACKEND.evict`, poll up to `EVICT_WAIT_S` (30 s), one retry; on success log `<name> is back in iCloud only` and clear `pulled`; on failure log `<name> is still on the Mac: macOS would not offload it` and keep `pulled`.

`cloud._sleep = time.sleep` is the only sleep, so tests replace it.

- [ ] **Step 1: Write the failing tests** in `tests/test_cloud.py` with a `FakeCloud` (a set of cloud-only paths; `download` discards, `evict` adds unless `refuse` is set; records calls) installed by a `cloud_env` fixture that sets `FILM_GAMES_DIR`, patches `cloud.BACKEND`, `cloud.is_cloud_only` and `cloud._sleep`. One test per rule:
  - `test_a_local_film_is_never_pulled_or_offloaded`
  - `test_a_cloud_only_film_is_pulled_then_put_back`
  - `test_nested_holds_offload_once_at_the_outermost_exit`
  - `test_another_live_process_holding_the_film_blocks_the_offload` (a hold file named for `os.getppid()`)
  - `test_a_dead_process_hold_is_ignored` (pid of a finished `subprocess.Popen`)
  - `test_too_little_room_raises_before_any_download` (patch `shutil.disk_usage`)
  - `test_a_failed_download_raises_and_leaves_no_hold`
  - `test_a_pull_that_never_finishes_times_out` (`FILM_CLOUD_TIMEOUT_S=4`, a `download` that does nothing)
  - `test_an_offload_that_does_not_take_is_logged_and_retried_by_the_next_job`
  - `test_the_film_goes_back_when_the_job_raises` (also `KeyboardInterrupt`)
  - `test_lazy_source_pulls_only_when_the_film_is_asked_for`
  - `test_film_state`
  - `test_is_cloud_only_reads_the_dataless_flag` (unpatched; a tmp file is not cloud-only, a missing file is not cloud-only)
- [ ] **Step 2: Run them, expect `ModuleNotFoundError: film.common.cloud`**
- [ ] **Step 3: Write `film/common/cloud.py`**
- [ ] **Step 4: Run `../../../.venv/bin/python -m pytest tests/test_cloud.py -q`, expect all pass**
- [ ] **Step 5: Commit** `Cloud-only films: pull on demand, hold across processes, put back when the last hold ends`

### Task 2: Wire the stages

**Files:**
- Modify: `film/ingest/run.py` (`ingest`), `film/perceive/run.py` (`perceive`), `film/interpret/run.py` (`ball_paths`, `picture_scores`, `classify_arrivals`), `film/commands/scoreboard.py` (`run`), `film/train/prelabel.py` (`prelabel_rims`, `autolabel_rims`, `prelabel_balls`), `film/train/rimclf_dataset.py` (`ensure_views`, `build_rimclf_candidates`), `film/export/clips.py` (`clips_for_game`), `film/commands/run.py` (`run`), `film/cli.py` (`main`)
- Test: `tests/test_cloud_wiring.py`

**Consumes:** `local_source`, `lazy_source`, `CloudError` from Task 1.

| Function | Change |
|---|---|
| `ingest` | body from `info = probe(source)` through profile detection inside `with local_source(source)` |
| `perceive` | the `iter_frames` loop inside `with local_source(Path(m.source), log=log)` |
| `classify_arrivals` | `with lazy_source(Path(m.source)) as film:` and pass `film` to `ball_paths` and `picture_scores`, which gain a keyword `film=None` defaulting to `lambda: Path(m.source)` and call `film()` at each read |
| `scoreboard.run` | `build_timeline` inside `with local_source(Path(m.source))` |
| `prelabel_rims`, `autolabel_rims`, `prelabel_balls` | frame loops inside `with local_source(Path(m.source), log=log)` |
| `ensure_views` | `with lazy_source(Path(m.source), log=log) as film:`, `film()` at the `read_rim_view` call |
| `build_rimclf_candidates` | `with lazy_source(Path(m.source), log=log) as film:` around the body through `ensure_views`; `source` becomes `film()` |
| `clips_for_game` | `with lazy_source(Path(m.source), log=log) as film:`, `film()` at `cut_clip`, so a game with nothing to cut pulls nothing |
| `commands/run.run` | everything after argument handling inside `with local_source(args.source)` |
| `cli.main` | `except CloudError as exc: print(f"error: {exc}", file=sys.stderr); return 2` around `args.func(args)` |

- [ ] **Step 1: Write failing tests** in `tests/test_cloud_wiring.py` using the Task 1 fixture (moved to `tests/conftest.py` as `cloud_env`):
  - `test_run_pulls_once_and_puts_back_once_across_all_stages` (stub perceive/interpret/clips as `tests/test_run_command.py` does; each stub takes its own `local_source`; assert `downloads == 1`, `evictions == 1`)
  - `test_ingest_pulls_a_cloud_only_film_and_puts_it_back`
  - `test_clips_with_nothing_to_cut_pulls_nothing`
  - `test_clips_pulls_when_a_clip_is_missing`
  - `test_a_cloud_error_exits_2_with_the_message` (`main(["ingest", ...])` with too little room)
- [ ] **Step 2: Run, expect failures on download counts**
- [ ] **Step 3: Make the changes in the table**
- [ ] **Step 4: Run the whole suite, expect all pass**
- [ ] **Step 5: Commit** `Every stage takes its film through the cloud hold`

### Task 3: Review server

**Files:**
- Create: `film/review/films.py`, `film/review/static/film.html`
- Modify: `film/review/app.py`, `film/review/static/index.html`
- Test: `tests/test_review_films.py`

**Produces:**
```python
class Films:                                   # film/review/films.py
    def __init__(self, idle_s: float | None = None, clock=time.monotonic)
    def status(self, game_id: str, source: Path) -> dict   # {"state": "local"|"pulling"|"cloud"|"missing", "size_bytes": int, "error": str|None}
    def request(self, game_id: str, source: Path) -> dict  # starts a pull if cloud-only; takes a hold if local; stamps last use; returns status
    def sweep(self) -> list[str]               # releases holds idle longer than idle_s; returns game ids released
    def release_all(self) -> None
```
A pull runs `cloud.hold` on a daemon thread; its `CloudError` message is kept and returned as `error` once, then cleared so the next request retries.

`app.py`: `create_app` builds `app.state.films = Films()`, a lifespan that starts a 60 s sweeper thread and calls `release_all()` on shutdown; `GET` and `POST /api/games/{game_id}/film` return `status` and `request`; `GET /video/{game_id}` calls `request` and returns `503` with `{"detail": "pulling the film from iCloud"}` unless the state is `local`.

`film.html`: reads `game` and `t` from the query string, `POST`s `/api/games/<game>/film`, shows `Pulling the full film from iCloud (<n.n> GB)…` and polls `GET` every 5 s, then `location.replace("/video/<game>#t=<t>")`. Shows the error text when one comes back.

`index.html`: the L key opens `/film.html?game=…&t=…`; `loadQuarters` requests the film first and, while it is not local, renders the same message and polls every 5 s before rendering the `<video>`.

- [ ] **Step 1: Write failing tests**: status of a local film; `request` on a cloud-only film returns `pulling` then `local` once the thread joins; `/video` returns 503 while cloud-only and 200 after; a pull error is reported once; `sweep` releases after `idle_s` on a fake clock and the film goes back; `release_all` puts every pulled film back; the existing `test_clip_and_ranged_video` still passes.
- [ ] **Step 2: Run, expect import failure**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the whole suite**
- [ ] **Step 5: Commit** `Review: the full film is pulled on first use and put back after 30 idle minutes`

### Task 4: Doctor and docs

**Files:**
- Modify: `film/commands/doctor.py`, `tests/test_doctor.py`, `docs/HANDOFF.md`, `README.md`

- [ ] **Step 1: Failing test** `test_doctor_lists_each_games_film_state`: ingest the synthetic video, expect a line `film local    2026-01-01-synthetic`.
- [ ] **Step 2: Implement**: after the paths block, one line per game with a manifest, `film {state:8s} {game_id}`.
- [ ] **Step 3: HANDOFF section "Films may be cloud-only"** and two README lines: any script that reads a film goes through `local_source`; never open `m.source` directly; `film doctor` shows where each film is.
- [ ] **Step 4: Run the whole suite**
- [ ] **Step 5: Commit** `Doctor lists where each film is; handoff says how to read one`

### Task 5: Real round trip (needs Garrett's go-ahead)

- [ ] With `FILM_GAMES_DIR` pointed at a scratch directory, offload `Arcadia vs Cortez - Angle 1.mp4` by hand (`brctl evict`), then read one frame from it inside `local_source` with a short Python snippet. Confirm the log shows the pull and the put-back and that the film is cloud-only afterwards. Then download it again so it ends in the state it started in.

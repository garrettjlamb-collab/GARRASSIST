# Rung 1 execution ledger (2026-09-30)

Kept from the subagent-driven execution of docs/superpowers/plans/2026-09-30-rung-1-shot-times.md. Lines marked `minor (deferred)` and `Parked` are the backlog for rung 2. Commit SHAs before the trailer rewrite are stale; subjects still match.

    # SDD ledger — plan: docs/superpowers/plans/2026-09-30-rung-1-shot-times.md
    Branch: rung-1-shot-times (created from main at 1422a77)
    Task 1: minor (deferred): tests/test_device.py never exercises cuda-only or mps-over-cuda priority
    Task 1: minor (deferred): doctor.py __import__ only catches ImportError; a native OSError would crash doctor
    Task 1: complete (commits 1422a77..8fc672e, review clean)
    Task 2: minor (deferred): manifest.py reads/writes without encoding="utf-8" and save() is not atomic (write-temp-then-os.replace)
    Task 2: minor (deferred): test_paths.py covers only FILM_GAMES_DIR, not the other three roots or the unset default; test_doctor depends on the shell env
    Task 2: complete (commits 8fc672e..e0f9114, review clean)
    Task 3: minor (deferred): probe() assumes a video stream with an average_rate and a non-None duration; iter_frames has no fps<=0 guard; no test for end<start or fractional start
    Task 3: note: commit trailer was injected as "Claude Sonnet 5.5" by the subagent harness and amended to Fable 5.1 (c95c12d -> 7cb1485); every later dispatch states the exact trailer
    Task 3: fix round 1/5 (1 addressed, 0 open — pixel-level seek assertion in test_iter_frames_window; commits 7cb1485..b8b6a85)
    Task 3: complete (commits e0f9114..b8b6a85, review clean)
    Task 4: ruling: reviewer's "led_fraction missing" was a typo in the plan's file map (no task consumes it); plan doc corrected in 3c6117d, code unchanged
    Task 4: minor (deferred): --profile is not validated against PROFILES; missing source raises a raw FileNotFoundError; warning printed to stdout not stderr; overlay region hardcoded to 1920x1080
    Task 4: note: objc "duplicate AVFFrameReceiver" warning when av and cv2 both load (bundled libavdevice in each); harmless so far, watch for crashes
    Task 4: fix round 1/5 (1 addressed by ruling, 1 open — template correlation added but digits drag it to 0.46 on real frames so CORR_MIN was lowered to 0.3; commits 18e469c..3116ce4)
    Task 4: fix round 2/5 (0 addressed, 1 open — masking digit cells alone left t1500 at 0.417 and t4800 at 0.395; cause is the court/crowd border around the panel inside the region; nothing committed)
    Task 4: fix round 3/5 (2 addressed, 0 open — mask restricted to the panel interior; Hudl 0.949-1.0 vs NFHS -0.071-0.108, CORR_MIN 0.6; commits 3116ce4..c2f8245)
    Task 4: minor (deferred): overlay_correlation on hudl_t2400 is tautological (template cropped from it); the other four frames carry the real signal
    Task 4: complete (commits b8b6a85..c2f8245, review clean)
    Task 5: note: implementer tuned JSON numbers only (clock sep_top -> [166,902,175,909], sep_bottom -> [167,926,174,930], score cells y 921 -> 922); zones and threshold untouched; sub-minute clock layout is handled by Task 6's read_clock (bottom dot only = SS.t)
    Task 5: minor (deferred): synthetic digit test is circular (paints SEGMENT_ZONES then decodes with them); real-frame asserts cover t1500 home/period/separators only until Task 6
    Task 5: minor (deferred): box_fraction does not guard a cell outside the frame (negative index wraps); DIGIT_PATTERNS omits tail-less 6/9 and 7-with-f variants (fail safe as -1); calibrate hardcodes a 1920x1080 zoom crop and lets an unknown --profile raise
    Task 5: minor (deferred): segment d on the t4800 "0" read 0.27 vs the 0.3 threshold before the 1-px y shift; no post-fix margin readout
    Task 5: complete (commits c2f8245..73ca29a, review clean)
    Task 6: pre-review fix requested — home score read None in 107/121 early reads; cause: HUE_RANGES["red"] stops at 12 but the home LEDs span hue 0-31; fix widens red to [(0,30),(150,180)] + fixture hudl_tactical_t60.jpg (home 0, away 0, clock 303, period 1)
    Task 6: ruling: red/amber hue overlap is by design (masks are read per-cell) so the selectivity test asserts exclusion of green and panel-blue only; SEGMENT_ZONES["g"] narrowed to (0.32,0.44,0.68,0.56) because the 20-px fouls cell read g=0.30 at t60 (a "0" became an "8"); Hudl fixture count bumped to 6
    Task 6: ruling: both Important findings are gaps in the plan's own code/tests (period-fall rule not recency-gated; jump cap and window edges untested); no plan text requires the defect, so they go to the fix loop, not the human
    Task 6: minor (deferred): read_clock turns a blank middle cell into 0 (1:05 with a dropped tens digit reads 0:05); fouls excluded from conf; has_scoreboard is true for any profile json; 12/121 early reads at conf 0.75 unexplained; g-zone margin 0.19 on the 20-px cells
    Task 6: fix round 1/5 (2 addressed, 0 open — period rule recency-gated; 16 synthetic tests for every cleaning rule and read_clock; commits baff0fa..7ac9a3b)
    Task 6: complete (commits 73ca29a..7ac9a3b, review clean)
    Task 7: minor (deferred): registry test lacks a v9/v10 ordering case; tracks round-trip test asserts no dtypes; open_vocab untested against a fake YOLOWorld; load_detector treats any non-rim kind as ball; WORLD_WEIGHTS is a bare filename so ultralytics downloads into cwd not models_dir(); write_manifest json.dumps has no default for Path/numpy values (Tasks 8-14 may hit this)
    Task 7: complete (commits 7ac9a3b..6e175a6, review clean)
    Task 8: minor (deferred): smooth_rims re-sorts same-frame rows by track_id (confidence order lost); window counts rows not frames across dropouts; greedy confidence-first matching; tests lack box-size threshold scaling, single-row smoothing, two-track smoothing
    Task 8: complete (commits 6e175a6..2fd5721, review clean)
    Task 9: minor (deferred): tests lack a right/bottom-edge rim, a two-rim call, and a crop-shape check
    Task 9: complete (commits 2fd5721..023c3de, review clean)
    Task 10: note: brief's synthetic blob (20x20) could not reach its own >5.0 assertion (3.82); test blob enlarged to 25x25, implementation unchanged; reviewer confirmed arithmetically
    Task 10: ruling: "stale tile on reappearance" promoted from minor to a fix round because the Tactical camera pans constantly and each re-entry would spike motion energy
    Task 10: minor (deferred): no test for a rim at a frame edge or motion away from the rim
    Task 10: fix round 1/5 (1 addressed, 0 open — max_gap_frames baseline reset on rim reappearance; commits ded6b2e..a292ea6)
    Task 10: complete (commits 023c3de..a292ea6, review clean)
    Env: CLIP added as a dependency (ee6f9f0); YOLO-World weights and CLIP ViT-B/32 cached; Python's default SSL store fails on downloads but certifi's works -> film CLI must set SSL_CERT_FILE to certifi (folded into Task 11)
    Finding: YOLO-World finds no rims in this film at any imgsz/prompt (returns the ball instead); YOLO-World runs ~46 fps on MPS at 640
    Task 22 (added): template rim detector for hudl-tactical; measured on 269 sampled frames: left 0.70-1.0 true vs <=0.64 false, right 0.73-0.98 true vs <0.70 false; plan text committed
    Task 22: minor (deferred): cross-template suppression untested; t3600/t4800 scores are self-matches (comment them as such); load_detector lost its return annotation
    Task 22: complete (commits 09e0628..7ecc0d6, review clean)
    Task 11: note: perceive runs 32-38 fps on the M3 (a full game in ~30 min); rims in 12% of the first 120 s is legitimate (baskets out of view during warmups; 53% of frames game-wide have one); SSL_CERT_FILE via certifi added to film/cli.py
    Task 11: minor (deferred): test_cli SSL test can leak SSL_CERT_FILE into later tests (delenv on an unset var records nothing); perceive test lacks limit_s, scoreboard-present path, smoothing, and stage-key coverage; empty-rim-rows path unexercised; scoreboard reads are first-frame-of-each-second
    Task 11: complete (commits 7ecc0d6..d7a0938, review clean)
    Task 12: minor (deferred): min-gap test uses overlapping frames (arc(0)+arc(8)) so the proof is muddy; per-rim independence, missing-rim reset, motion local-max/min-gap, short-group skip, boundary and merge tie order untested; unused fps parameter
    Task 12: complete (commits d7a0938..2d67b55, review clean)
    Env: torchvision resnet18 ImageNet weights cached at ~/.cache/torch/hub/checkpoints (downloaded with SSL_CERT_FILE=certifi)
    Task 13: minor (deferred): RimClassifier.load happy path untested; train resizes via PIL bilinear while predict uses cv2 INTER_AREA (small shift; share one path); missing split dir raises FileNotFoundError not ValueError; epochs=0 writes no best.pt
    Task 13: complete (commits 2d67b55..d7657a5, review clean)
    Task 14: ruling: "ambiguous scoreboard counts as a miss" is a gap in the plan's rules (no plan text requires it) -> fix loop: three-way verdict made/missed/unknown/ambiguous, ambiguous = classifier alone + flag scoreboard-ambiguous
    Task 14: minor (deferred): clf-none without scoreboard guesses missed instead of None; score window floor 0.5 s can overlap the next arrival's lead read; tests lack team=away, no-classifier missed, other-rim window, and >3 cases; event_id round-half-even at .x5
    Env: yolo11n.pt base weights cached in the repo root (git-ignored) for Task 16
    Task 14: fix round 1/5 (1 addressed, 0 open — three-way scoreboard verdict, ambiguous = no evidence + flag; commits 45dd01c..9420f49)
    Task 14: complete (commits d7657a5..9420f49, review clean)
    Task 15: ruling: both Important findings enter the fix loop (yolo_line clamping is in the plan's own code but no plan text requires the defect); split becomes a pure function of the image id
    Task 15: minor (deferred): prelabel_balls truncates [:n] after flattening (biases to early events); no-rim events skipped silently (fixed in round 1 via no_rim count); rim_box_at arbitrary on two rims per frame; imwrite return unchecked; pending JSON stores absolute image paths; data.yaml path unquoted; --n 0 becomes the default
    Task 15: fix round 1/5 (2 addressed, 0 open — clamped labels, crc32 id-based splits, idempotent accepts, rerun-safe candidates; commits 4ce893e..0ea8f9c)
    Task 15: minor (deferred): reclassifying an id that was a class's only val image leaves val without that class until the next store
    Task 15: complete (commits 9420f49..0ea8f9c, review clean)
    Task 16: minor (deferred): a training run that raises midway leaves an orphan vN folder without best.pt (invisible to latest(), leaves a numbered gap and debris); refusal tests lack match= and val-only case; _metric None path and no-weights RuntimeError untested; --epochs 0 becomes the default
    Task 16: complete (commits 0ea8f9c..7361991, review clean)
    Task 17: ruling: three Important findings (swallowed ffmpeg stderr, over-broad mpeg4 fallback, partial clip skipped as done) are in the plan's code but nothing in the plan requires them -> fix loop
    Task 17: minor (deferred): build_reel silently drops unreadable clips, never checks isOpened, and does not release the writer on exception; reel test does not check labels or total frame count; no --force test
    Task 17: fix round 1/5 (3 addressed, 0 open — ffmpeg stderr surfaced, encoder-only fallback, part-file rename; commits bd77103..49914ad)
    Task 17: complete (commits 7361991..49914ad, review clean)
    Task 18: ruling: three Important findings (suffix ranges wrong, non-atomic racy review.json, unvalidated dataset/cls on label routes) + the Starlette httpx warning -> fix loop; httpx2 replaces httpx as the dev dependency
    Task 18: minor (deferred): /video 500s when the source file is missing (should be 404); tests do not check review.json on disk, dataset files after accept, or range body bytes; events re-parsed per request
    Task 18: fix round 1/5 (4 addressed, 0 open — suffix ranges, locked atomic review.json, validated label routes, httpx2; commits d6349fc..d24a066)
    Task 18: minor (deferred): no explicit test that re-accepting an accepted id returns 200; path params unsanitised (local tool); _known_image globs pending per request
    Task 18: complete (commits 49914ad..d24a066, review clean)
    Task 19: note: during the controller's browser check three decision POSTs appeared that the controller did not trigger, at the same time the Task 19 reviewer subagent (which has browser tools) was running on the same tab; the agent-made review.json was deleted and the key check re-run in isolation
    Task 19: controller browser check (chrome-devtools) on the real game: page renders, clip plays, M toggles the current card, Enter posts the decision with the coach's made value and advances, sidebar count updates, review.json written; Label rim tab renders the 1920x1080 frame on the canvas and proposes nothing for a frame with no basket (correct)
    Task 19: ruling: three Important findings (no ev.repeat guard, no error handling, drag inside an existing box selects it) are in the plan's JavaScript but nothing in the plan requires them -> fix loop; favicon 404 folded in
    Task 19: minor (deferred): no input-focus guard on keydown; tab switches racy while loadLabel is in flight; pos() includes the 1px canvas border; review command's Timer still opens the browser if the port is taken; the test only checks strings in the HTML
    Task 19: fix round 1/5 (4 addressed, 0 open — ev.repeat + busy guard, status line with try/catch, click-vs-drag with smallest-box select, favicon; commits 25d6c99..ef6442f)
    Task 19: controller re-check after the fix: clean reload with no console errors, status element present, undecided event sorted first, U key posts an unsure decision
    Task 19: minor (deferred): J/K/M/T not guarded while a decide POST is in flight; state.drawing not cleared on a new image; click on empty canvas now deselects; dead `const pos` line in renderBoxLabel; fix-round report shows only summaries of the test runs
    Task 19: complete (commits d24a066..ef6442f, review clean)
    Task 20: minor (deferred): reel test does not prove confirmed-only or the None return; precision/made_agreement None cases unasserted; an explicit "made": None decision counts as disagreement; H:MM:SS reference lines raise a bare ValueError
    Task 20: complete (commits ef6442f..e6ffd0d, review clean)
    Task 21: controller deleted the test-made review.json and the three export files for the real game before the full run so Garrett starts clean
    Task 21: full game on fallback models: 1788 s wall clock, 54040 frames, 28324 rim rows, 242 ball rows, 5309 scoreboard reads, 201 arrivals (200 motion, 1 ball), 201 shot events (13 made, 158 missed, 30 undecided), 199 clips; the untrained ball detector is the weak link
    Task 21: minor (deferred): unused `import types` in tests/test_run_command.py; README numbered list still puts rim training first
    Task 21: complete (commits e6ffd0d..713d265, review clean)
    All tasks complete (1-22). Final whole-branch review next.
    Final review (fable, 1422a77..713d265): no Critical; 12 Important (dropped makes, same-id score window, per-id motion threshold, unknown events never export, prelabel rimclf ignores review.json, no refusal on unknown profile, --profile unvalidated, bare weight filenames, trained rim outranks templates, one broken game dir blanks /api/games, Manifest.save not atomic, no run idempotence); verdict "merge with fixes"; one fix wave follows
    Fix wave: agent cut off by a rate limit after A1-A5, B1-B5, C1 (12 commits f9b63b4..bfe616c, suite 168 green); resumed for C2-C6 and the report
    Fix wave: all 12 commits carry the injected trailer "Claude Opus 5.5"; controller will rewrite the trailers on the fix range with git filter-branch --msg-filter once the agent is done
    Fix wave: real game after section B (interpret re-run 14:02): 258 events = 225 motion + 32 scoreboard-only + 1 ball; made 44 (was 13), missed 189, None 25; no duplicate point credit
    Fix wave: complete — 18 commits f9b63b4..8af5091 (A1-A5, B1-B5, C1-C6, plus a test fix that stopped two tests moving the repo's real weights into pytest temp dirs); suite 180 passed, no warnings; D1 re-perceive running in the background; scoped re-review dispatched on 713d265..8af5091; trailers still to be rewritten from "Claude Opus 5.5" to "Claude Fable 5.1" once the agent hands back
    Fix-wave re-review (opus): 15/16 addressed; C4 PARTIAL — a --limit-s smoke run marks perceive done and the next full run skips it (Important, load-bearing for the README workflow); one bounded follow-up dispatched for it plus minors: gap-aware smooth_rims, pending->class promotion in prelabel rimclf, distinct-frame rim warning
    Parked with ruling: "rise dropped by both paths" (latent; 0 cases on the real game; rung 2 should mark a rise explained only by an event whose scoreboard verdict is made); B2 residual under 1 s between arrivals (0 cases); evaluate scores type==shot only; decisions freeze made/team at decision time; load_decisions/load_timeline without encoding

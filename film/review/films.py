"""The review server's claims on full films.

The cards play pre-cut clips, so the full film is only wanted for the Quarters tab and the L key. A film
that is in iCloud only is pulled on a background thread while the page waits, and goes back after a
stretch with nobody watching it.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from film.common import cloud


class Films:
    def __init__(self, idle_s: float | None = None, clock=time.monotonic, log=print):
        self.idle_s = float(os.environ.get("FILM_CLOUD_IDLE_S", "1800")) if idle_s is None else idle_s
        self._clock, self._log = clock, log
        self._lock = threading.Lock()
        self._holds: dict[str, cloud.Hold] = {}
        self._used: dict[str, float] = {}
        self._pulls: dict[str, threading.Thread] = {}
        self._errors: dict[str, str] = {}

    def status(self, game_id: str, source: Path) -> dict:
        source = Path(source)
        with self._lock:
            pulling, error = game_id in self._pulls, self._errors.get(game_id)
        if not source.exists():
            state = "missing"
        elif pulling:
            state = "pulling"
        else:
            state = "cloud" if cloud.is_cloud_only(source) else "local"
        return {"state": state, "size_bytes": source.stat().st_size if state != "missing" else 0, "error": error}

    def request(self, game_id: str, source: Path) -> dict:
        """Somebody wants this film: stamp it as in use, and start pulling it if it is in iCloud only."""
        source = Path(source)
        stale = None
        with self._lock:
            self._used[game_id] = self._clock()
            if game_id not in self._pulls and source.exists():
                self._errors.pop(game_id, None)               # asking again is how a failed pull is retried
                if cloud.is_cloud_only(source):
                    stale = self._holds.pop(game_id, None)    # the coach offloaded it while we held it
                    thread = threading.Thread(target=self._pull, args=(game_id, source), daemon=True)
                    self._pulls[game_id] = thread
                    thread.start()
                elif game_id not in self._holds:
                    self._holds[game_id] = cloud.hold(source, log=self._log)
        if stale is not None:
            stale.release()
        return self.status(game_id, source)

    def touch(self, game_id: str) -> None:
        with self._lock:
            self._used[game_id] = self._clock()

    def _pull(self, game_id: str, source: Path) -> None:
        held, error = None, None
        try:
            held = cloud.hold(source, log=self._log)
        except Exception as exc:  # noqa: BLE001 - whatever stopped the pull is shown on the page
            error = str(exc)
        with self._lock:
            self._pulls.pop(game_id, None)
            self._used[game_id] = self._clock()
            if held is not None:
                self._holds[game_id] = held
            else:
                self._errors[game_id] = error

    def wait(self, timeout: float | None = None) -> None:
        """Block until no pull is running."""
        with self._lock:
            threads = list(self._pulls.values())
        for thread in threads:
            thread.join(timeout)

    def sweep(self) -> list[str]:
        """Let go of every film nobody has watched for idle_s; a film this server pulled goes back to iCloud."""
        now = self._clock()
        with self._lock:
            idle = [g for g in self._holds if now - self._used.get(g, now) >= self.idle_s]
            holds = [self._holds.pop(g) for g in idle]
        for held in holds:
            held.release()
        return idle

    def release_all(self) -> None:
        with self._lock:
            holds = list(self._holds.values())
            self._holds.clear()
        for held in holds:
            held.release()

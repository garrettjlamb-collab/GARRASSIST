"""Films that live in iCloud Drive with no local copy.

A job pulls its film down when it needs it and puts it back when the last job using it ends. Only a film
the pipeline pulled is ever offloaded: one the coach keeps on the Mac stays on the Mac.

Any code that reads a film goes through `local_source` (or `lazy_source` when the film is only sometimes
read); nothing opens a manifest's `source` directly.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from film.common.paths import games_dir

SF_DATALESS = 0x40000000          # st_flags of a file whose contents are in iCloud only; Python 3.12 has no name for it
POLL_S = 2.0
REPORT_EVERY_S = 30.0
EVICT_WAIT_S = 30.0
EVICT_RETRY_S = 5.0

_sleep = time.sleep
_clock = time.monotonic


class CloudError(RuntimeError):
    """A film could not be brought down from iCloud. The message is written for the coach."""


class Backend:
    """The two macOS commands that move an iCloud Drive file in and out."""

    def download(self, path: Path) -> None:
        done = subprocess.run(["brctl", "download", str(path)], capture_output=True, text=True)
        if done.returncode != 0:
            raise CloudError(f"could not pull {Path(path).name} from iCloud: {(done.stderr or done.stdout).strip()}")

    def evict(self, path: Path) -> bool:
        """True when macOS accepted the request. It refuses a file that has not finished uploading."""
        return subprocess.run(["brctl", "evict", str(path)], capture_output=True, text=True).returncode == 0


BACKEND = Backend()


def is_cloud_only(path: Path) -> bool:
    try:
        return bool(getattr(os.stat(path), "st_flags", 0) & SF_DATALESS)
    except OSError:
        return False


def _state_dir(source: Path) -> Path:
    key = hashlib.sha1(str(Path(source).resolve()).encode()).hexdigest()[:16]
    return games_dir() / ".cloud" / key


def film_state(source: Path) -> str:
    """missing, cloud (in iCloud only), pulled (on the Mac because the pipeline pulled it), or local."""
    source = Path(source)
    if not source.exists():
        return "missing"
    if is_cloud_only(source):
        return "cloud"
    return "pulled" if (_state_dir(source) / "pulled").exists() else "local"


@contextmanager
def _locked(state: Path):
    (state / "holds").mkdir(parents=True, exist_ok=True)
    with open(state / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _live_holds(state: Path) -> list[Path]:
    """The holds of running processes; a hold left by a process that has died is deleted."""
    live = []
    for file in (state / "holds").iterdir():
        try:
            alive = _alive(int(file.name.split("-")[0]))
        except ValueError:
            alive = False
        if alive:
            live.append(file)
        else:
            file.unlink(missing_ok=True)
    return live


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def _pull(source: Path, log) -> None:
    size = source.stat().st_size
    cushion = float(os.environ.get("FILM_CLOUD_CUSHION_GB", "5")) * 1e9
    free = shutil.disk_usage(source.parent).free
    if free < size + cushion:
        raise CloudError(f"not enough room to pull {source.name} from iCloud: it is {_gb(size)} and the disk has "
                         f"{_gb(free)} free ({_gb(cushion)} is kept clear)")
    timeout = float(os.environ.get("FILM_CLOUD_TIMEOUT_S", "3600"))
    log(f"pulling {source.name} ({_gb(size)}) from iCloud")
    BACKEND.download(source)
    started, next_report = _clock(), REPORT_EVERY_S
    while is_cloud_only(source):
        waited = _clock() - started
        if waited >= timeout:
            raise CloudError(f"{source.name} did not arrive from iCloud in {timeout / 60:.0f} minutes; "
                             f"check the connection and run it again")
        if waited >= next_report:
            log(f"  still pulling {source.name}: {next_report / 60:.1f} min")
            next_report += REPORT_EVERY_S
        _sleep(POLL_S)


def _offload(source: Path) -> bool:
    """Ask macOS to drop the local copy and see that it did: a request it accepts does not always take."""
    for attempt in range(2):
        if attempt:
            _sleep(EVICT_RETRY_S)
        if not BACKEND.evict(source):
            continue
        started = _clock()
        while True:
            if is_cloud_only(source):
                return True
            if _clock() - started >= EVICT_WAIT_S:
                break
            _sleep(1.0)
    return False


class Hold:
    """One job's claim on a film. While any hold is live the film is not put back."""

    def __init__(self, source: Path, file: Path | None, log):
        self.source, self._file, self._log = source, file, log

    def release(self) -> None:
        file, self._file = self._file, None
        if file is None:
            return
        state = file.parent.parent
        with _locked(state):
            file.unlink(missing_ok=True)
            pulled = state / "pulled"
            if _live_holds(state) or not pulled.exists():
                return
            if is_cloud_only(self.source) or not self.source.exists():      # the coach already moved it
                pulled.unlink(missing_ok=True)
            elif _offload(self.source):
                pulled.unlink(missing_ok=True)
                self._log(f"{self.source.name} is back in iCloud only")
            else:
                # The mark stays, so the next job on this film tries again when it ends.
                self._log(f"{self.source.name} is still on the Mac: macOS would not offload it")


def hold(source: Path, log=print) -> Hold:
    """Make the film local, pulling it from iCloud if need be, and keep it local until release()."""
    source = Path(source)
    state = _state_dir(source)
    pulled = state / "pulled"
    if not is_cloud_only(source) and not pulled.exists():
        return Hold(source, None, log)            # a film nobody will ever offload needs no claim
    with _locked(state):
        _live_holds(state)
        file = state / "holds" / f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        file.write_text("")
        if is_cloud_only(source):
            marked = pulled.exists()
            # Marked before the download starts: a job that finds the film local a moment later must
            # see that it is on loan and take its own hold.
            pulled.write_text("")
            try:
                _pull(source, log)
            except BaseException:
                file.unlink(missing_ok=True)
                if not marked:
                    pulled.unlink(missing_ok=True)
                raise
    return Hold(source, file, log)


@contextmanager
def local_source(source: Path, log=print) -> Iterator[Path]:
    """The film, on the Mac for as long as the block runs."""
    held = hold(source, log=log)
    try:
        yield Path(source)
    finally:
        held.release()


@contextmanager
def lazy_source(source: Path, log=print) -> Iterator[Callable[[], Path]]:
    """A function that returns the film, pulling it on the first call: for a stage that often has
    everything it needs on disk already and should not pull 5 GB to find that out."""
    held: list[Hold] = []

    def film() -> Path:
        if not held:
            held.append(hold(source, log=log))
        return Path(source)

    try:
        yield film
    finally:
        for h in held:
            h.release()

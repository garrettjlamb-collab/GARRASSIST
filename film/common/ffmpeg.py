from __future__ import annotations

import shutil


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: the system one if present, else the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

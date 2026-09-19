"""FastF1 disk cache setup.

Risk mitigation for fastF1 data volume & rate limits: always enable the
cache before pulling anything. Re-running a script (e.g. after
`download_historical.py` is interrupted) then costs nothing for rounds
already fetched.
"""
from __future__ import annotations

from pathlib import Path

import fastf1

# repo_root/data/cache
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def enable_cache(cache_dir: Path | None = None) -> Path:
    path = cache_dir or CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(path))
    return path

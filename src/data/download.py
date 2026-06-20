"""Download the freMTPL2 dataset (French Motor Third-Party Liability).

    freMTPL2freq : one row per policy (risk features + claim count + exposure)
    freMTPL2sev  : one row per claim  (policy id + claim amount)

Fetched from OpenML and cached as CSV under data/raw/. Internet is needed on the
first run only; afterwards the cached CSVs are reused.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OPENML_FREQ_ID = 41214
OPENML_SEV_ID = 41215


def download_fremtpl2(raw_dir: str | Path) -> tuple[Path, Path]:
    """Return (freq_csv_path, sev_csv_path), downloading + caching if needed."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    freq_path = raw_dir / "fremtpl2_freq.csv"
    sev_path = raw_dir / "fremtpl2_sev.csv"

    if freq_path.exists() and sev_path.exists():
        logger.info("freMTPL2 already cached in %s", raw_dir)
        return freq_path, sev_path

    from sklearn.datasets import fetch_openml  # imported lazily so tests don't need it

    logger.info("Downloading freMTPL2freq (OpenML id=%s)...", OPENML_FREQ_ID)
    freq = fetch_openml(data_id=OPENML_FREQ_ID, as_frame=True).frame
    logger.info("Downloading freMTPL2sev (OpenML id=%s)...", OPENML_SEV_ID)
    sev = fetch_openml(data_id=OPENML_SEV_ID, as_frame=True).frame

    # IDpol comes back as float or string depending on version; normalise to int.
    freq["IDpol"] = freq["IDpol"].astype("int64")
    sev["IDpol"] = sev["IDpol"].astype("int64")

    freq.to_csv(freq_path, index=False)
    sev.to_csv(sev_path, index=False)
    logger.info("Saved %d policy rows and %d claim rows.", len(freq), len(sev))
    return freq_path, sev_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_fremtpl2("data/raw")

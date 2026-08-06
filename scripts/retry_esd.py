"""Retry-loop the ESD download until all 7 parquet shards are present."""

import time
from pathlib import Path

from huggingface_hub import snapshot_download


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ESD_DATA = RAW_DIR / "esd" / "data"
NEEDED = 7


def complete() -> bool:
    return len(list(ESD_DATA.glob("train-*.parquet"))) >= NEEDED


def main() -> None:
    for attempt in range(60):
        if complete():
            print("ESD complete:", len(list(ESD_DATA.glob("train-*.parquet"))), "shards", flush=True)
            return
        try:
            snapshot_download(
                repo_id="jspaulsen/esd",
                repo_type="dataset",
                allow_patterns=["data/*"],
                local_dir=RAW_DIR / "esd",
            )
            print("snapshot_download returned", flush=True)
        except Exception as e:
            print(f"attempt {attempt}: {type(e).__name__} {str(e)[:120]}", flush=True)
        time.sleep(45)
    print("gave up after 60 attempts", flush=True)


if __name__ == "__main__":
    main()

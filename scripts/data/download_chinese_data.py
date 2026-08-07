"""Download Chinese speech emotion datasets (ESD + CSEMOTIONS) from HuggingFace."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_esd() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = snapshot_download(
        repo_id="jspaulsen/esd",
        repo_type="dataset",
        allow_patterns=["data/*"],
        local_dir=RAW_DIR / "esd",
    )
    print("ESD downloaded ->", out)
    return Path(out)


def download_csemotions() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = snapshot_download(
        repo_id="sdcsdccdsd/CSEMOTIONS",
        repo_type="dataset",
        allow_patterns=["data/*"],
        local_dir=RAW_DIR / "csemotions",
    )
    print("CSEMOTIONS downloaded ->", out)
    return Path(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["esd", "csemotions"], default=None)
    args = parser.parse_args()
    if args.only in (None, "esd"):
        download_esd()
    if args.only in (None, "csemotions"):
        download_csemotions()


if __name__ == "__main__":
    main()

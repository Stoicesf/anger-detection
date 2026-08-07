"""Download RAVDESS speech audio and CREMA-D audio."""

import argparse
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_ravdess_speech() -> Path:
    """RAVDESS speech-only audio (1440 wav, ~215 MB) from a GitHub mirror."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = (
        "https://codeload.github.com/ZenvilleErasmus/"
        "RAVDESS-emotions-speech-audio-only/zip/refs/heads/master"
    )
    out_zip = RAW_DIR / "RAVDESS_speech.zip"
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(out_zip, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\rRAVDESS: {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()
    print(f"RAVDESS downloaded -> {out_zip}")
    return out_zip


def download_cremad() -> Path:
    """CREMA-D full dataset mirror (audio + metadata)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = snapshot_download(
        repo_id="myleslinder/crema-d",
        repo_type="dataset",
        local_dir=RAW_DIR / "crema-d",
    )
    print(f"CREMA-D downloaded -> {out_dir}")
    return Path(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["ravdess", "cremad"], default=None)
    args = parser.parse_args()

    if args.only in (None, "ravdess"):
        download_ravdess_speech()
    if args.only in (None, "cremad"):
        download_cremad()


if __name__ == "__main__":
    main()

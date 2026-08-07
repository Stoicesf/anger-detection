"""Shared paths and constants for the anger detection pipeline."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk up until VERSION or pyproject.toml is found."""
    cur = (start or Path(__file__).resolve()).parent
    for p in [cur, *cur.parents]:
        if (p / "VERSION").exists() or (p / "pyproject.toml").exists():
            return p
    # fallback: src/anger_detection -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CURATED_DIR = DATA_DIR / "curated"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reports"

# numba cache must live in a writable location
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


SR = 16000          # target sample rate
N_FFT = 480         # 30 ms window
HOP_LENGTH = 160    # 10 ms hop
N_MELS = 40         # mel filter count
TIME_FRAMES = 98    # fixed time dimension

NUM_CLASSES = 5
CLASS_NAMES = [
    "0_中性 (neutral)",
    "1_高兴 (happy)",
    "2_愤怒 (angry)",
    "3_悲伤 (sad)",
    "4_惊讶 (surprise)",
]

# RAVDESS
RAVDESS_SPEECH_DIR = (
    RAW_DIR / "ravdess_speech" / "RAVDESS-emotions-speech-audio-only-master" / "Audio_Speech_Actors_01-24"
)
# CREMA-D
CREMAD_DIR = RAW_DIR / "crema-d" / "data" / "AudioWAV"
# 中文数据集
ESD_DIR = RAW_DIR / "esd" / "data"
CSEMOTIONS_DIR = RAW_DIR / "csemotions" / "data"


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, CURATED_DIR, MODELS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def env_path(name: str, default: str) -> Path:
    """Resolve an external dataset root from env, with a Windows-friendly default."""
    return Path(os.environ.get(name, default))


CASIA_ROOT = env_path("ANGER_CASIA_ROOT", r"E:\桌面\test\CASIA")
CNEV_ROOT = env_path(
    "ANGER_CNEV_ROOT",
    r"E:\桌面\test\CNEV_Vocalizations\Core Set",
)


def resolve_curated_path(rel: str | Path) -> Path:
    """Map legacy `dataset/...` manifest paths onto `data/curated/...`."""
    p = Path(rel)
    if p.is_absolute():
        # rewrite absolute paths that still point at old dataset/ layout under this repo
        try:
            rel_to_root = p.relative_to(PROJECT_ROOT)
            return resolve_curated_path(rel_to_root)
        except ValueError:
            return p
    parts = p.as_posix().replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "dataset":
        return CURATED_DIR.joinpath(*parts[1:])
    if parts and parts[0] == "data" and len(parts) >= 2 and parts[1] == "curated":
        return PROJECT_ROOT.joinpath(*parts)
    return PROJECT_ROOT / p

"""Shared paths and constants for the anger detection pipeline."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

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
RAVDESS_SPEECH_DIR = RAW_DIR / "ravdess_speech" / "RAVDESS-emotions-speech-audio-only-master" / "Audio_Speech_Actors_01-24"
# CREMA-D
CREMAD_DIR = RAW_DIR / "crema-d" / "data" / "AudioWAV"
# 中文数据集
ESD_DIR = RAW_DIR / "esd" / "data"
CSEMOTIONS_DIR = RAW_DIR / "csemotions" / "data"


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

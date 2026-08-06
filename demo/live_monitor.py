"""Background microphone stream + anger inference (shared across Streamlit reruns)."""
from __future__ import annotations

import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import numpy as np


@dataclass
class LiveConfig:
    sr: int = 16000
    window_s: float = 1.5
    hop_s: float = 0.5
    anger_threshold: float = 0.55
    clear_threshold: float = 0.35
    require_argmax: bool = True
    min_rms: float = 0.005


@dataclass
class LiveSnapshot:
    running: bool = False
    is_angry: bool = False
    anger_p: float = 0.0
    rms: float = 0.0
    detail: str = "未启动"
    lat_ms: float = 0.0
    updated_at: float = 0.0
    frames: int = 0
    probs: Any = None
    history: list = field(default_factory=list)
    error: str = ""


class LiveAngerMonitor:
    """Continuous mic capture + ONNX anger detection."""

    ANGRY_IDX = 2

    def __init__(self):
        self._lock = threading.Lock()
        self._cfg = LiveConfig()
        self._snap = LiveSnapshot()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buf = deque(maxlen=int(16000 * 4))  # ~4s ring
        self._stream = None
        self._sess = None
        self._inp_name = None
        self._predict_fn = None  # (audio_float32) -> (probs, lat_s)

    def update_config(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._cfg, k):
                    setattr(self._cfg, k, v)

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            h = list(self._snap.history)
            return LiveSnapshot(
                running=self._snap.running,
                is_angry=self._snap.is_angry,
                anger_p=self._snap.anger_p,
                rms=self._snap.rms,
                detail=self._snap.detail,
                lat_ms=self._snap.lat_ms,
                updated_at=self._snap.updated_at,
                frames=self._snap.frames,
                probs=None if self._snap.probs is None else np.array(self._snap.probs),
                history=h,
                error=self._snap.error,
            )

    def start(self, predict_fn) -> None:
        """predict_fn(audio: np.ndarray) -> (probs: np.ndarray, latency_s: float)"""
        if self._thread and self._thread.is_alive():
            return
        self._predict_fn = predict_fn
        self._stop.clear()
        with self._lock:
            self._snap = LiveSnapshot(running=True, detail="实时监听中…")
            self._buf.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        with self._lock:
            self._snap.running = False
            if not self._snap.detail.startswith("错误"):
                self._snap.detail = "已停止"

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            pass
        mono = indata[:, 0].astype(np.float32, copy=True)
        with self._lock:
            self._buf.extend(mono.tolist())

    def _decide(self, probs: np.ndarray, prev: bool, cfg: LiveConfig) -> tuple[bool, float, str]:
        anger_p = float(probs[self.ANGRY_IDX])
        top = int(np.argmax(probs))
        if prev:
            if anger_p >= cfg.clear_threshold:
                return True, anger_p, "保持愤怒"
            return False, anger_p, "愤怒已解除"
        ok = anger_p >= cfg.anger_threshold
        if cfg.require_argmax:
            ok = ok and top == self.ANGRY_IDX
        if ok:
            return True, anger_p, "检测到愤怒"
        return False, anger_p, "常规（非愤怒）"

    def _run(self) -> None:
        import sounddevice as sd

        try:
            with self._lock:
                cfg0 = LiveConfig(**self._cfg.__dict__)
            self._stream = sd.InputStream(
                samplerate=cfg0.sr,
                channels=1,
                dtype="float32",
                blocksize=int(cfg0.sr * 0.1),
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            with self._lock:
                self._snap.running = False
                self._snap.error = f"麦克风打开失败: {e}"
                self._snap.detail = f"错误: {e}"
            return

        try:
            while not self._stop.is_set():
                with self._lock:
                    cfg = LiveConfig(**self._cfg.__dict__)
                    need = int(cfg.sr * cfg.window_s)
                    samples = list(self._buf)
                    prev_angry = self._snap.is_angry

                if len(samples) < need:
                    time.sleep(0.05)
                    continue

                audio = np.asarray(samples[-need:], dtype=np.float32)
                rms = float(np.sqrt((audio**2).mean()))

                if rms < cfg.min_rms:
                    anger_p = 0.0
                    is_angry = False
                    detail = "静音 → 非愤怒"
                    lat_ms = 0.0
                    probs = None
                else:
                    try:
                        probs, lat = self._predict_fn(audio)
                        is_angry, anger_p, detail = self._decide(probs, prev_angry, cfg)
                        lat_ms = lat * 1000
                    except Exception as e:
                        with self._lock:
                            self._snap.error = str(e)
                            self._snap.detail = f"推理错误: {e}"
                        time.sleep(cfg.hop_s)
                        continue

                with self._lock:
                    changed = is_angry != self._snap.is_angry
                    self._snap.is_angry = is_angry
                    self._snap.anger_p = anger_p
                    self._snap.rms = rms
                    self._snap.detail = detail
                    self._snap.lat_ms = lat_ms
                    self._snap.updated_at = time.time()
                    self._snap.frames += 1
                    self._snap.probs = probs
                    self._snap.running = True
                    if changed or is_angry:
                        self._snap.history.insert(
                            0,
                            {
                                "t": time.strftime("%H:%M:%S"),
                                "label": "愤怒" if is_angry else "非愤怒",
                                "angry": is_angry,
                                "anger_p": anger_p,
                                "detail": detail,
                                "lat_ms": lat_ms,
                                "rms": rms,
                            },
                        )
                        self._snap.history = self._snap.history[:40]

                time.sleep(max(cfg.hop_s, 0.15))
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None
            with self._lock:
                self._snap.running = False


def audio_to_temp_wav(audio: np.ndarray, sr: int) -> Path:
    path = Path(gettempdir()) / "anger_live_chunk.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        w.writeframes(pcm.tobytes())
    return path

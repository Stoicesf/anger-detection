"""Background microphone stream + anger inference (shared across Streamlit reruns).

Phase-1 decision layer: EMA + consecutive hits + NORMAL/SUSPECT/ANGER/RECOVER.
"""
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

from decision import (
    AlarmState,
    AngerStateMachine,
    DecisionConfig,
    STATE_LABEL_ZH,
)


@dataclass
class LiveConfig:
    sr: int = 16000
    window_s: float = 1.0
    hop_s: float = 0.4
    anger_threshold: float = 0.70
    clear_threshold: float = 0.35
    require_argmax: bool = True
    min_rms: float = 0.005
    ema_alpha: float = 0.30
    required_hits: int = 3
    recover_hold_s: float = 3.0
    recover_display_s: float = 1.0


@dataclass
class LiveSnapshot:
    running: bool = False
    is_angry: bool = False
    state: str = AlarmState.NORMAL.value
    state_zh: str = STATE_LABEL_ZH[AlarmState.NORMAL]
    anger_p: float = 0.0  # smoothed score (primary)
    raw_anger_p: float = 0.0
    hit_count: int = 0
    required_hits: int = 3
    rms: float = 0.0
    detail: str = "未启动"
    lat_ms: float = 0.0
    updated_at: float = 0.0
    frames: int = 0
    probs: Any = None
    history: list = field(default_factory=list)
    error: str = ""


class LiveAngerMonitor:
    """Continuous mic capture + ONNX anger detection + event decision."""

    ANGRY_IDX = 2

    def __init__(self):
        self._lock = threading.Lock()
        self._cfg = LiveConfig()
        self._snap = LiveSnapshot()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buf = deque(maxlen=int(16000 * 4))
        self._stream = None
        self._predict_fn = None
        self._sm = AngerStateMachine(self._decision_cfg())

    def _decision_cfg(self) -> DecisionConfig:
        c = self._cfg
        return DecisionConfig(
            ema_alpha=c.ema_alpha,
            threshold=c.anger_threshold,
            clear_threshold=c.clear_threshold,
            required_hits=c.required_hits,
            require_argmax=c.require_argmax,
            recover_hold_s=c.recover_hold_s,
            recover_display_s=c.recover_display_s,
        )

    def update_config(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._cfg, k):
                    setattr(self._cfg, k, v)
            self._sm.configure(
                ema_alpha=self._cfg.ema_alpha,
                threshold=self._cfg.anger_threshold,
                clear_threshold=self._cfg.clear_threshold,
                required_hits=self._cfg.required_hits,
                require_argmax=self._cfg.require_argmax,
                recover_hold_s=self._cfg.recover_hold_s,
                recover_display_s=self._cfg.recover_display_s,
            )

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            h = list(self._snap.history)
            return LiveSnapshot(
                running=self._snap.running,
                is_angry=self._snap.is_angry,
                state=self._snap.state,
                state_zh=self._snap.state_zh,
                anger_p=self._snap.anger_p,
                raw_anger_p=self._snap.raw_anger_p,
                hit_count=self._snap.hit_count,
                required_hits=self._snap.required_hits,
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
        if self._thread and self._thread.is_alive():
            return
        self._predict_fn = predict_fn
        self._stop.clear()
        with self._lock:
            self._sm = AngerStateMachine(self._decision_cfg())
            self._snap = LiveSnapshot(
                running=True,
                detail="实时监听中…",
                required_hits=self._cfg.required_hits,
            )
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

    def reset_state(self) -> None:
        with self._lock:
            self._sm.reset()
            self._snap.is_angry = False
            self._snap.state = AlarmState.NORMAL.value
            self._snap.state_zh = STATE_LABEL_ZH[AlarmState.NORMAL]
            self._snap.anger_p = 0.0
            self._snap.raw_anger_p = 0.0
            self._snap.hit_count = 0
            self._snap.detail = "已手动重置"
            self._snap.probs = None

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        mono = indata[:, 0].astype(np.float32, copy=True)
        with self._lock:
            self._buf.extend(mono.tolist())

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

                if len(samples) < need:
                    time.sleep(0.05)
                    continue

                audio = np.asarray(samples[-need:], dtype=np.float32)
                rms = float(np.sqrt((audio**2).mean()))
                silent = rms < cfg.min_rms
                now = time.time()
                probs = None
                lat_ms = 0.0
                raw = 0.0
                top_idx = None

                if not silent:
                    try:
                        probs, lat = self._predict_fn(audio)
                        lat_ms = lat * 1000
                        raw = float(probs[self.ANGRY_IDX])
                        top_idx = int(np.argmax(probs))
                    except Exception as e:
                        with self._lock:
                            self._snap.error = str(e)
                            self._snap.detail = f"推理错误: {e}"
                        time.sleep(cfg.hop_s)
                        continue

                with self._lock:
                    out = self._sm.update(
                        raw, top_idx=top_idx, now=now, silent=silent
                    )
                    prev_state = self._snap.state
                    self._snap.is_angry = out.is_alarm
                    self._snap.state = out.state.value
                    self._snap.state_zh = STATE_LABEL_ZH[out.state]
                    self._snap.anger_p = out.smooth_score
                    self._snap.raw_anger_p = out.raw_score
                    self._snap.hit_count = out.hit_count
                    self._snap.required_hits = cfg.required_hits
                    self._snap.rms = rms
                    self._snap.detail = out.detail
                    self._snap.lat_ms = lat_ms
                    self._snap.updated_at = now
                    self._snap.frames += 1
                    self._snap.probs = probs
                    self._snap.running = True

                    changed = prev_state != self._snap.state
                    if changed or out.is_alarm:
                        self._snap.history.insert(
                            0,
                            {
                                "t": time.strftime("%H:%M:%S"),
                                "label": self._snap.state_zh,
                                "state": self._snap.state,
                                "angry": out.is_alarm,
                                "anger_p": out.smooth_score,
                                "raw_p": out.raw_score,
                                "hits": out.hit_count,
                                "detail": out.detail,
                                "lat_ms": lat_ms,
                                "rms": rms,
                            },
                        )
                        self._snap.history = self._snap.history[:50]

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

"""Phase-1 acceptance metrics on Chinese audio (FAR / detection delay).

Simulates streaming inference with the Phase-1 decision layer.

Examples:
  python scripts/eval_realtime_metrics.py
  python scripts/eval_realtime_metrics.py --audio-dir "E:\\桌面\\test\\CASIA"
  python scripts/eval_realtime_metrics.py --long-negative-minutes 30

Language: Chinese speech only (CASIA / CNEV).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort


from anger_detection.common import (
    HOP_LENGTH,
    MODELS_DIR,
    N_FFT,
    N_MELS,
    OUTPUT_DIR,
    SR,
    TIME_FRAMES,
    CASIA_ROOT,
)  # noqa: E402
from anger_detection.decision import AngerStateMachine, DecisionConfig, AlarmState  # noqa: E402
from anger_detection.features import log_mel_full  # noqa: E402

CASIA = CASIA_ROOT
ANGRY_IDX = 2


def pick_onnx() -> Path:
    for p in (
        MODELS_DIR / "dscnn_casia_deploy.onnx",
        MODELS_DIR / "dscnn_casia.onnx",
        MODELS_DIR / "dscnn.onnx",
    ):
        if p.exists():
            return p
    raise FileNotFoundError("no onnx model")


def center_crop(feat: np.ndarray) -> np.ndarray:
    t = feat.shape[0]
    if t >= TIME_FRAMES:
        s = (t - TIME_FRAMES) // 2
        return feat[s : s + TIME_FRAMES]
    pb = (TIME_FRAMES - t) // 2
    return np.pad(feat, ((pb, TIME_FRAMES - t - pb), (0, 0)), mode="edge")


class OnnxAnger:
    def __init__(self, path: Path):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def predict_wav_array(self, y: np.ndarray) -> tuple[float, int, float]:
        """Return (anger_prob, top_idx, latency_s) from float waveform."""
        # Write-free mel: reuse log_mel pipeline via temp is slow; compute here
        t0 = time.perf_counter()
        rms = float(np.sqrt((y**2).mean()))
        if rms > 1e-4:
            y = y * (0.05 / rms)
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        db = np.clip(librosa.power_to_db(mel, ref=1.0), -60.0, 20.0)
        feat = ((db + 60.0) / 80.0).T.astype(np.float32)
        patch = center_crop(feat)
        x = patch.reshape(1, 1, TIME_FRAMES, 40).astype(np.float32)
        probs = self.sess.run(None, {self.inp: x})[0][0]
        lat = time.perf_counter() - t0
        return float(probs[ANGRY_IDX]), int(np.argmax(probs)), lat


def load_mono(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def list_casia(emotion: str) -> list[Path]:
    d = CASIA / emotion
    return sorted(d.glob("*.wav")) if d.is_dir() else []


def stream_decide(
    y: np.ndarray,
    model: OnnxAnger,
    sm: AngerStateMachine,
    *,
    window_s: float,
    hop_s: float,
    min_rms: float,
) -> dict:
    """Run decision SM over a long waveform; return alarm events & latency stats."""
    win = int(window_s * SR)
    hop = int(hop_s * SR)
    sm.reset()
    t_audio = 0.0
    now = 0.0
    events = []
    latencies = []
    in_alarm = False
    first_alarm_audio_t = None
    n_frames = 0

    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))

    for start in range(0, len(y) - win + 1, hop):
        chunk = y[start : start + win]
        rms = float(np.sqrt((chunk**2).mean()))
        silent = rms < min_rms
        if silent:
            raw, top = 0.0, None
            lat = 0.0
        else:
            raw, top, lat = model.predict_wav_array(chunk)
            latencies.append(lat)
        now += hop_s
        t_audio = (start + win) / SR
        out = sm.update(raw, top_idx=top, now=now, silent=silent)
        n_frames += 1
        if out.is_alarm and not in_alarm:
            in_alarm = True
            if first_alarm_audio_t is None:
                first_alarm_audio_t = t_audio
            events.append({"t": t_audio, "type": "ALARM_ON", "score": out.smooth_score})
        elif (not out.is_alarm) and in_alarm:
            in_alarm = False
            events.append({"t": t_audio, "type": "ALARM_OFF", "score": out.smooth_score})

    duration_s = len(y) / SR
    alarm_ons = [e for e in events if e["type"] == "ALARM_ON"]
    return {
        "duration_s": duration_s,
        "duration_h": duration_s / 3600.0,
        "n_frames": n_frames,
        "n_false_alarms": len(alarm_ons),  # for pure-negative streams
        "events": events,
        "first_alarm_s": first_alarm_audio_t,
        "lat_mean_ms": float(np.mean(latencies) * 1000) if latencies else 0.0,
        "lat_p50_ms": float(np.percentile(latencies, 50) * 1000) if latencies else 0.0,
    }


def build_long_negative(minutes: float, seed: int = 42, emotions=None) -> np.ndarray:
    """Concatenate Chinese non-angry CASIA clips to ~minutes duration."""
    rng = np.random.default_rng(seed)
    if emotions is None:
        emotions = ("neutral", "happy", "sad", "surprise", "fear")
    pools = []
    for emo in emotions:
        pools.extend(list_casia(emo))
    rng.shuffle(pools)
    target = int(minutes * 60 * SR)
    chunks = []
    total = 0
    i = 0
    while total < target and pools:
        p = pools[i % len(pools)]
        y = load_mono(p)
        g = float(rng.uniform(0.6, 1.8))
        chunks.append(y * g)
        gap = np.zeros(int(SR * float(rng.uniform(0.15, 0.6))), dtype=np.float32)
        chunks.append(gap)
        total += len(y) + len(gap)
        i += 1
    y = np.concatenate(chunks) if chunks else np.zeros(target, dtype=np.float32)
    return y[:target]


def eval_detection_delay(model: OnnxAnger, cfg: DecisionConfig, window_s: float, hop_s: float, min_rms: float) -> dict:
    """For each angry file: delay from start to first ANGER state."""
    delays = []
    misses = 0
    for wav in list_casia("angry"):
        y = load_mono(wav)
        sm = AngerStateMachine(cfg)
        r = stream_decide(y, model, sm, window_s=window_s, hop_s=hop_s, min_rms=min_rms)
        if r["first_alarm_s"] is None:
            misses += 1
        else:
            # anger present from start of clip → delay ≈ first_alarm_s - window_s
            delays.append(max(0.0, r["first_alarm_s"] - window_s))
    return {
        "n": len(list_casia("angry")),
        "misses": misses,
        "detect_rate": 1.0 - misses / max(len(list_casia("angry")), 1),
        "delay_mean_s": float(np.mean(delays)) if delays else None,
        "delay_p50_s": float(np.percentile(delays, 50)) if delays else None,
        "delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-negative-minutes", type=float, default=20.0)
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--hop-s", type=float, default=0.4)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--ema-alpha", type=float, default=0.30)
    ap.add_argument("--required-hits", type=int, default=3)
    ap.add_argument("--clear-threshold", type=float, default=0.35)
    ap.add_argument("--recover-hold-s", type=float, default=3.0)
    ap.add_argument("--min-rms", type=float, default=0.005)
    ap.add_argument("--skip-delay", action="store_true")
    args = ap.parse_args()

    onnx_path = pick_onnx()
    print("model:", onnx_path)
    model = OnnxAnger(onnx_path)
    cfg = DecisionConfig(
        ema_alpha=args.ema_alpha,
        threshold=args.threshold,
        clear_threshold=args.clear_threshold,
        required_hits=args.required_hits,
        require_argmax=True,
        recover_hold_s=args.recover_hold_s,
    )

    print(f"\n[1/3] FAR on Chinese low-arousal (neutral+sad) ~{args.long_negative_minutes} min …")
    y_low = build_long_negative(args.long_negative_minutes, emotions=("neutral", "sad"))
    sm = AngerStateMachine(cfg)
    far_low = stream_decide(
        y_low, model, sm, window_s=args.window_s, hop_s=args.hop_s, min_rms=args.min_rms
    )
    far_low_h = far_low["n_false_alarms"] / max(far_low["duration_h"], 1e-9)
    print(f"  FAR_low: {far_low_h:.2f}/hour (alarms={far_low['n_false_alarms']})")

    print(f"\n[2/3] FAR on Chinese hard-negative mix (happy/surprise/fear/neutral) …")
    y_neg = build_long_negative(args.long_negative_minutes)
    sm = AngerStateMachine(cfg)
    far = stream_decide(
        y_neg,
        model,
        sm,
        window_s=args.window_s,
        hop_s=args.hop_s,
        min_rms=args.min_rms,
    )
    far_per_hour = far["n_false_alarms"] / max(far["duration_h"], 1e-9)
    print(
        f"  FAR_hard: {far['n_false_alarms']} alarms / {far['duration_h']:.3f} h "
        f"= {far_per_hour:.2f} /hour"
    )
    print(f"  infer latency p50={far['lat_p50_ms']:.2f} ms")

    delay = None
    if not args.skip_delay:
        print("\n[3/3] Detection delay on CASIA angry (Chinese) …")
        delay = eval_detection_delay(
            model, cfg, args.window_s, args.hop_s, args.min_rms
        )
        print(
            f"  detect_rate={delay['detect_rate']:.3f} misses={delay['misses']}/{delay['n']}"
        )
        print(
            f"  delay mean/p50/p95 = {delay['delay_mean_s']:.2f}/"
            f"{delay['delay_p50_s']:.2f}/{delay['delay_p95_s']:.2f} s"
        )

    out = {
        "language": "zh-CN",
        "model": str(onnx_path),
        "decision": {
            "ema_alpha": args.ema_alpha,
            "threshold": args.threshold,
            "required_hits": args.required_hits,
            "window_s": args.window_s,
            "hop_s": args.hop_s,
            "recover_hold_s": args.recover_hold_s,
        },
        "far_low_arousal": {
            "duration_h": far_low["duration_h"],
            "n_false_alarms": far_low["n_false_alarms"],
            "far_per_hour": far_low_h,
            "pass": far_low_h < 1.0,
        },
        "far": {
            "duration_h": far["duration_h"],
            "n_false_alarms": far["n_false_alarms"],
            "far_per_hour": far_per_hour,
            "lat_p50_ms": far["lat_p50_ms"],
            "target_far_per_hour": 1.0,
            "pass": far_per_hour < 1.0,
            "note": "includes high-arousal non-anger (happy/surprise) — harder",
        },
        "delay": delay,
        "targets": {"far_per_hour": 1.0, "delay_s": 2.0},
    }
    if delay and delay["delay_mean_s"] is not None:
        out["delay"]["pass"] = delay["delay_mean_s"] < 2.0

    out_path = OUTPUT_DIR / "realtime_metrics_zh.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved", out_path)
    print(
        "RESULT:",
        "FAR",
        "PASS" if out["far"]["pass"] else "FAIL",
        f"({far_per_hour:.2f}/h)",
        "| delay",
        (
            "PASS"
            if delay and delay.get("pass")
            else ("FAIL" if delay else "SKIP")
        ),
    )


if __name__ == "__main__":
    main()

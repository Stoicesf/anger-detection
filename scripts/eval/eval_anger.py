"""Evaluate binary anger model + Day3-A threshold/hits grid + LOSO calibration.

Usage:
  python scripts/eval/eval_anger.py --onnx artifacts/models/dscnn_anger_v2.onnx --grid
  python scripts/eval/eval_anger.py --onnx artifacts/models/dscnn_anger_v2.onnx --realtime --grid-thr --grid-hits
  python scripts/eval/eval_anger.py --onnx artifacts/models/dscnn_anger_v2.onnx --grid --calib-T
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


from anger_detection.common import (  # noqa: E402
    HOP_LENGTH,
    MODELS_DIR,
    N_FFT,
    N_MELS,
    OUTPUT_DIR,
    SR,
    TIME_FRAMES,
    CASIA_ROOT,
)
from anger_detection.decision import AngerStateMachine, DecisionConfig  # noqa: E402

CASIA = CASIA_ROOT
FOLDER_POS = {"angry", "anger"}
FOLDER_NEG = {"neutral", "happy", "happiness", "sad", "sadness", "surprise", "fear"}

THR_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
RT_THR_GRID = [0.45, 0.50, 0.55, 0.60, 0.65]
HITS_GRID = [1, 2, 3, 4]
T_GRID = [0.5, 0.7, 1.0, 1.5, 2.0]


def center_crop(feat: np.ndarray) -> np.ndarray:
    t = feat.shape[0]
    if t >= TIME_FRAMES:
        s = (t - TIME_FRAMES) // 2
        return feat[s : s + TIME_FRAMES]
    pb = (TIME_FRAMES - t) // 2
    return np.pad(feat, ((pb, TIME_FRAMES - t - pb), (0, 0)), mode="edge")


def wav_to_mel(y: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt((y**2).mean()))
    if rms > 1e-4:
        y = y * (0.05 / rms)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    db = np.clip(librosa.power_to_db(mel, ref=1.0), -60.0, 20.0)
    return ((db + 60.0) / 80.0).T.astype(np.float32)


def apply_temperature(prob: float | np.ndarray, T: float) -> float | np.ndarray:
    """ONNX already has sigmoid; invert -> logit/T -> sigmoid."""
    if T == 1.0:
        return prob
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    out = 1.0 / (1.0 + np.exp(-logit / T))
    if np.isscalar(prob):
        return float(out)
    return out.astype(np.float32)


class OnnxBinaryAnger:
    def __init__(self, path: Path):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        out_shape = self.sess.get_outputs()[0].shape
        self.binary = True
        if len(out_shape) >= 2 and out_shape[-1] not in (1, "1", None):
            try:
                if int(out_shape[-1]) == 5:
                    self.binary = False
            except (TypeError, ValueError):
                pass

    def score_array(self, y: np.ndarray) -> float:
        feat = wav_to_mel(y.astype(np.float32))
        patch = center_crop(feat)
        x = patch.reshape(1, 1, TIME_FRAMES, N_MELS).astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0][0]
        if self.binary:
            return float(np.asarray(out).reshape(-1)[0])
        return float(out[2])

    def predict(self, y: np.ndarray) -> tuple[float, float]:
        t0 = time.perf_counter()
        return self.score_array(y), time.perf_counter() - t0


def speaker_of(path: Path) -> str:
    return path.stem.split("-")[-1]


def collect_casia() -> list[dict]:
    rows = []
    for folder in sorted(CASIA.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if name in FOLDER_POS:
            lab = 1
        elif name in FOLDER_NEG:
            lab = 0
        else:
            continue
        for wav in sorted(folder.glob("*.wav")):
            rows.append(
                {
                    "path": wav,
                    "label": lab,
                    "speaker": speaker_of(wav),
                    "emotion": name,
                }
            )
    return rows


def metrics_at_thr(yt: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    yp = (scores >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    return {
        "thr": float(thr),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fp": fp,
        "fn": fn,
        "tp": int(cm[1, 1]),
        "tn": int(cm[0, 0]),
    }


def precompute_file_scores(model: OnnxBinaryAnger, rows: list[dict]) -> dict:
    """Center-crop score per file (for classification / LOSO / T grids)."""
    yt, scores, speakers, emotions, durs = [], [], [], [], []
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        y, _ = librosa.load(row["path"], sr=SR, mono=True)
        scores.append(model.score_array(y))
        yt.append(row["label"])
        speakers.append(row["speaker"])
        emotions.append(row["emotion"])
        durs.append(len(y) / SR)
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  file-scores {i+1}/{len(rows)}", flush=True)
    print(f"  file-scores done in {time.perf_counter()-t0:.1f}s", flush=True)
    return {
        "yt": np.asarray(yt, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float32),
        "speakers": np.asarray(speakers, dtype=object),
        "emotions": np.asarray(emotions, dtype=object),
        "durs": np.asarray(durs, dtype=np.float32),
    }


def precompute_stream_scores(
    model: OnnxBinaryAnger,
    rows: list[dict],
    *,
    window_s: float,
    hop_s: float,
) -> list[dict]:
    """Per-file list of (t, score) for offline decision sweep."""
    win = int(window_s * SR)
    hop = int(hop_s * SR)
    out = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        y, _ = librosa.load(row["path"], sr=SR, mono=True)
        frames = []
        for start in range(0, max(1, len(y) - win + 1), hop):
            chunk = y[start : start + win]
            if len(chunk) < win // 2:
                break
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            frames.append((start / SR, model.score_array(chunk)))
        if not frames and len(y) > 0:
            chunk = y if len(y) >= win else np.pad(y, (0, win - len(y)))
            frames.append((0.0, model.score_array(chunk[:win])))
        out.append(
            {
                "label": row["label"],
                "speaker": row["speaker"],
                "emotion": row["emotion"],
                "duration_s": len(y) / SR,
                "frames": frames,  # list[(t, score)]
            }
        )
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"  stream-scores {i+1}/{len(rows)}", flush=True)
    print(f"  stream-scores done in {time.perf_counter()-t0:.1f}s", flush=True)
    return out


def run_decision_on_stream(
    stream_rows: list[dict],
    *,
    thr: float,
    hits: int,
    ema: float,
    T: float = 1.0,
    clear_threshold: float | None = None,
) -> dict:
    clear = 0.35 if clear_threshold is None else clear_threshold
    # keep clear below thr so recover is reachable
    clear = min(clear, thr * 0.55)
    cfg = DecisionConfig(
        ema_alpha=ema,
        threshold=thr,
        clear_threshold=clear,
        required_hits=hits,
        require_argmax=False,
        recover_hold_s=5.0,
        recover_display_s=1.0,
    )
    delays = []
    detected = 0
    pos_n = 0
    false_alarms = 0
    neg_hours = 0.0
    fp_by_emotion: dict[str, int] = {}

    for row in stream_rows:
        lab = row["label"]
        sm = AngerStateMachine(cfg)
        if lab == 1:
            pos_n += 1
        else:
            neg_hours += row["duration_s"] / 3600.0

        fired = False
        for t, raw in row["frames"]:
            score = float(apply_temperature(raw, T))
            out = sm.update(score, top_idx=None, now=t, silent=False)
            if lab == 1 and out.is_alarm:
                detected += 1
                delays.append(t)
                fired = True
                break
            if lab == 0 and out.is_alarm:
                false_alarms += 1
                emo = row["emotion"]
                fp_by_emotion[emo] = fp_by_emotion.get(emo, 0) + 1
                fired = True
                break
        _ = fired

    far = false_alarms / max(neg_hours, 1e-9)
    return {
        "thr": thr,
        "hits": hits,
        "ema": ema,
        "T": T,
        "detect": detected / max(pos_n, 1),
        "detected": detected,
        "pos_n": pos_n,
        "false_alarms": false_alarms,
        "neg_hours": neg_hours,
        "FAR_per_hour": far,
        "mean_delay_s": float(np.mean(delays)) if delays else None,
        "median_delay_s": float(np.median(delays)) if delays else None,
        "fp_by_emotion": fp_by_emotion,
    }


def score_meets_home(row: dict) -> float:
    """Home priority: FAR<1/h, then Recall>85%, then Delay<2s.

    Prefer satisfying FAR first, but among FAR-failing rows still
    reward high detect so the Pareto front is visible.
    """
    far = row["FAR_per_hour"]
    det = row["detect"]
    delay = row["mean_delay_s"] if row["mean_delay_s"] is not None else 99.0
    ok_far = far <= 1.0
    ok_rec = det >= 0.85
    ok_delay = delay <= 2.0
    if ok_far and ok_rec and ok_delay:
        return 3000 + det * 10 - delay
    if ok_far and ok_delay:
        return 2000 + det * 100  # maximize recall under FAR budget
    if ok_rec and ok_delay:
        return 1000 - min(far, 200) + det * 10
    return det * 10 - min(far, 100) * 0.5 - min(delay, 10) * 0.2


def grid_classification(cache: dict, thrs: list[float], T: float = 1.0) -> list[dict]:
    scores = apply_temperature(cache["scores"], T)
    yt = cache["yt"]
    rows = []
    for thr in thrs:
        m = metrics_at_thr(yt, scores, thr)
        # file-level "FAR proxy": FP / total_neg_hours (using file durations)
        neg_mask = yt == 0
        neg_hours = float(cache["durs"][neg_mask].sum() / 3600.0)
        m["FAR_proxy_per_hour"] = m["fp"] / max(neg_hours, 1e-9)
        m["T"] = T
        rows.append(m)
    return rows


def grid_loso(
    cache: dict, speaker: str, thrs: list[float], T: float = 1.0
) -> list[dict]:
    mask = cache["speakers"] == speaker
    yt = cache["yt"][mask]
    scores = apply_temperature(cache["scores"][mask], T)
    rows = []
    for thr in thrs:
        m = metrics_at_thr(yt, scores, thr)
        m["speaker"] = speaker
        m["n"] = int(mask.sum())
        m["n_pos"] = int(yt.sum())
        m["T"] = T
        rows.append(m)
    return rows


def find_thr_for_recall(rows: list[dict], target_r: float = 0.85) -> dict | None:
    """Lowest thr is not wanted; pick highest thr that still meets recall."""
    ok = [r for r in rows if r["recall"] >= target_r]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["thr"], r["precision"], r["f1"]))


def print_table(headers: list[str], rows: list[list], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        cells = [f"{c:.3f}" if isinstance(c, float) else str(c) for c in row]
        str_rows.append(cells)
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for cells in str_rows:
        print(fmt.format(*cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(MODELS_DIR / "dscnn_anger_v2.onnx"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--rt-thr", type=float, default=0.65)
    ap.add_argument("--rt-hits", type=int, default=4)
    ap.add_argument("--rt-ema", type=float, default=0.25)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--test-speaker", default="ZhaoZuoxiang")
    ap.add_argument("--grid", action="store_true", help="full Day3-A grids")
    ap.add_argument("--grid-thr", action="store_true")
    ap.add_argument("--grid-hits", action="store_true")
    ap.add_argument("--calib-T", action="store_true", help="temperature scaling sweep")
    ap.add_argument("--neg-limit", type=int, default=400, help="neg files for stream grid")
    args = ap.parse_args()

    do_grid = args.grid or args.grid_thr or args.grid_hits or args.calib_T
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"missing onnx: {onnx_path}")

    model = OnnxBinaryAnger(onnx_path)
    rows = collect_casia()
    print(f"model={onnx_path.name} binary={model.binary} files={len(rows)}")

    # Always cache center-crop scores once
    print("Precompute center-crop scores…")
    cache = precompute_file_scores(model, rows)
    yt, scores = cache["yt"], cache["scores"]
    auc = float(roc_auc_score(yt, scores)) if len(np.unique(yt)) > 1 else None
    prauc = float(average_precision_score(yt, scores)) if len(np.unique(yt)) > 1 else None
    print(f"AUC={auc:.4f} PR-AUC={prauc:.4f}")

    result: dict = {
        "onnx": str(onnx_path),
        "auc": auc,
        "pr_auc": prauc,
        "n": len(rows),
        "test_speaker": args.test_speaker,
    }

    # single-thr classification
    cls = metrics_at_thr(yt, scores, args.thr)
    cls.update({"auc": auc, "pr_auc": prauc, "threshold": args.thr})
    print("=== Classification @ thr={:.2f} ===".format(args.thr))
    print(json.dumps(cls, indent=2))
    result["classification"] = cls

    if do_grid or args.grid_thr or args.grid:
        thr_rows = grid_classification(cache, THR_GRID, T=1.0)
        result["thr_grid_casia"] = thr_rows
        print_table(
            ["thr", "P", "R", "F1", "FP", "FN", "FAR_proxy/h"],
            [
                [
                    r["thr"],
                    r["precision"],
                    r["recall"],
                    r["f1"],
                    r["fp"],
                    r["fn"],
                    r["FAR_proxy_per_hour"],
                ]
                for r in thr_rows
            ],
            "CASIA threshold grid (center crop)",
        )

        loso_rows = grid_loso(cache, args.test_speaker, THR_GRID, T=1.0)
        result["thr_grid_loso"] = loso_rows
        print_table(
            ["thr", "P", "R", "F1", "FP", "FN"],
            [
                [r["thr"], r["precision"], r["recall"], r["f1"], r["fp"], r["fn"]]
                for r in loso_rows
            ],
            f"LOSO threshold grid ({args.test_speaker})",
        )
        pick = find_thr_for_recall(loso_rows, 0.85)
        if pick:
            print(
                f"\n>> LOSO thr for Recall>=85%: thr={pick['thr']:.2f} "
                f"P={pick['precision']:.3f} R={pick['recall']:.3f} F1={pick['f1']:.3f}"
            )
            result["loso_thr_recall85"] = pick
        else:
            print("\n>> No thr reaches LOSO Recall>=85% in grid (try lower thr / calib-T)")

        # score distribution diagnostics
        sp_mask = cache["speakers"] == args.test_speaker
        other = ~sp_mask
        pos_other = scores[(yt == 1) & other]
        pos_loso = scores[(yt == 1) & sp_mask]
        neg_all = scores[yt == 0]
        dist = {
            "pos_other_mean": float(pos_other.mean()) if len(pos_other) else None,
            "pos_other_p50": float(np.median(pos_other)) if len(pos_other) else None,
            "pos_loso_mean": float(pos_loso.mean()) if len(pos_loso) else None,
            "pos_loso_p50": float(np.median(pos_loso)) if len(pos_loso) else None,
            "neg_mean": float(neg_all.mean()),
            "neg_p95": float(np.percentile(neg_all, 95)),
            "neg_p99": float(np.percentile(neg_all, 99)),
        }
        result["score_dist"] = dist
        print("\n=== Score distribution ===")
        print(json.dumps(dist, indent=2))

    if args.calib_T or args.grid:
        t_summary = []
        for T in T_GRID:
            loso = grid_loso(cache, args.test_speaker, THR_GRID, T=T)
            pick = find_thr_for_recall(loso, 0.85)
            casia_at = None
            if pick:
                casia_at = metrics_at_thr(
                    yt, apply_temperature(scores, T), pick["thr"]
                )
            t_summary.append(
                {
                    "T": T,
                    "loso_best": pick,
                    "casia_at_loso_thr": casia_at,
                }
            )
        result["temperature_sweep"] = t_summary
        print_table(
            ["T", "loso_thr", "loso_P", "loso_R", "casia_P", "casia_R", "casia_FP"],
            [
                [
                    s["T"],
                    s["loso_best"]["thr"] if s["loso_best"] else "-",
                    s["loso_best"]["precision"] if s["loso_best"] else "-",
                    s["loso_best"]["recall"] if s["loso_best"] else "-",
                    s["casia_at_loso_thr"]["precision"] if s["casia_at_loso_thr"] else "-",
                    s["casia_at_loso_thr"]["recall"] if s["casia_at_loso_thr"] else "-",
                    s["casia_at_loso_thr"]["fp"] if s["casia_at_loso_thr"] else "-",
                ]
                for s in t_summary
            ],
            "Temperature scaling (LOSO Recall>=85% thr)",
        )

    need_rt = args.realtime or args.grid or args.grid_hits
    if need_rt:
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        rng = np.random.default_rng(42)
        if len(neg) > args.neg_limit:
            idx = rng.choice(len(neg), args.neg_limit, replace=False)
            neg = [neg[i] for i in idx]
        stream_subset = pos + neg
        print(
            f"Precompute stream scores: pos={len(pos)} neg={len(neg)} "
            f"window={args.window}s hop={args.hop}s"
        )
        stream = precompute_stream_scores(
            model, stream_subset, window_s=args.window, hop_s=args.hop
        )

        if args.grid or args.grid_hits or args.grid_thr:
            thr_list = RT_THR_GRID if (args.grid or args.grid_thr or args.grid_hits) else [args.rt_thr]
            hits_list = HITS_GRID if (args.grid or args.grid_hits) else [args.rt_hits]
            rt_grid = []
            for thr in thr_list:
                for hits in hits_list:
                    m = run_decision_on_stream(
                        stream,
                        thr=thr,
                        hits=hits,
                        ema=args.rt_ema,
                        T=1.0,
                    )
                    m["home_score"] = score_meets_home(m)
                    rt_grid.append(m)
            result["rt_grid"] = rt_grid
            print_table(
                ["thr", "hits", "detect", "FAR/h", "delay", "FA", "home"],
                [
                    [
                        r["thr"],
                        r["hits"],
                        r["detect"],
                        r["FAR_per_hour"],
                        r["mean_delay_s"] if r["mean_delay_s"] is not None else -1,
                        r["false_alarms"],
                        r["home_score"],
                    ]
                    for r in rt_grid
                ],
                f"Realtime thr×hits (EMA={args.rt_ema})",
            )
            ranked = sorted(rt_grid, key=lambda r: r["home_score"], reverse=True)
            best = ranked[0]
            print(
                f"\n>> Recommended (home priority): thr={best['thr']} hits={best['hits']} "
                f"detect={best['detect']:.3f} FAR/h={best['FAR_per_hour']:.2f} "
                f"delay={best['mean_delay_s']}"
            )
            result["recommended"] = best

            # also evaluate expected presets
            for name, thr, hits in [
                ("user_guess_55_2", 0.55, 2),
                ("user_guess_60_3", 0.60, 3),
                ("old_65_4", 0.65, 4),
            ]:
                m = run_decision_on_stream(
                    stream, thr=thr, hits=hits, ema=args.rt_ema, T=1.0
                )
                print(
                    f"  preset {name}: detect={m['detect']:.3f} "
                    f"FAR/h={m['FAR_per_hour']:.2f} delay={m['mean_delay_s']} "
                    f"fp_emo={m['fp_by_emotion']}"
                )
                result.setdefault("presets", {})[name] = m
        else:
            rt = run_decision_on_stream(
                stream,
                thr=args.rt_thr,
                hits=args.rt_hits,
                ema=args.rt_ema,
            )
            print("=== Realtime ===")
            print(json.dumps(rt, indent=2))
            result["realtime"] = rt

    out = OUTPUT_DIR / "eval_anger_v2_grid.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # also write a compact markdown table
    md_lines = ["# Anger V2 Day3-A Grid", ""]
    if "thr_grid_casia" in result:
        md_lines += ["## CASIA thr", "", "| thr | P | R | F1 | FP | FN |", "|-----|---|---|----|----|----|"]
        for r in result["thr_grid_casia"]:
            md_lines.append(
                f"| {r['thr']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['fp']} | {r['fn']} |"
            )
        md_lines.append("")
    if "thr_grid_loso" in result:
        md_lines += [
            f"## LOSO ({args.test_speaker})",
            "",
            "| thr | P | R | F1 | FP | FN |",
            "|-----|---|---|----|----|----|",
        ]
        for r in result["thr_grid_loso"]:
            md_lines.append(
                f"| {r['thr']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['fp']} | {r['fn']} |"
            )
        md_lines.append("")
    if "rt_grid" in result:
        md_lines += [
            "## Realtime thr×hits",
            "",
            "| thr | hits | detect | FAR/h | delay | FA |",
            "|-----|------|--------|-------|-------|----|",
        ]
        for r in result["rt_grid"]:
            d = f"{r['mean_delay_s']:.2f}" if r["mean_delay_s"] is not None else "-"
            md_lines.append(
                f"| {r['thr']:.2f} | {r['hits']} | {r['detect']:.3f} | "
                f"{r['FAR_per_hour']:.2f} | {d} | {r['false_alarms']} |"
            )
        md_lines.append("")
    if "recommended" in result:
        b = result["recommended"]
        md_lines += [
            "## Recommended",
            "",
            f"- thr={b['thr']} hits={b['hits']} EMA={args.rt_ema}",
            f"- detect={b['detect']:.3f} FAR/h={b['FAR_per_hour']:.2f} delay={b['mean_delay_s']}",
            "",
        ]
    md_path = OUTPUT_DIR / "eval_anger_v2_grid.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print("wrote", out)
    print("wrote", md_path)


if __name__ == "__main__":
    main()

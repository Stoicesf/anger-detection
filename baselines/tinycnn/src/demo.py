"""
TinyCNN 愤怒检测 · 实时演示界面

Usage:
  conda activate pytorch12
  cd e:\\桌面\\TinyCNN
  python src/demo.py
  # 浏览器打开 http://127.0.0.1:7861
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature import DURATION, HOP_LENGTH, N_FFT, N_MFCC, SR, extract_mfcc
from model import TinyCNN

CKPT_PATH = ROOT / "checkpoints" / "best_model.pth"
# 常规/临界状态判为非愤怒；略提高阈值降低日常误报
DEFAULT_THRESHOLD = 0.65
DEFAULT_SCORES = {"非愤怒": 1.0, "愤怒": 0.0}


def load_model(ckpt_path: Path = CKPT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 1)
    model = TinyCNN(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "best_epoch": ckpt.get("epoch"),
        "val_f1": ckpt.get("val_f1"),
        "val_acc": ckpt.get("val_acc"),
        "in_channels": in_channels,
    }
    return model, device, meta


MODEL, DEVICE, META = load_model()


def _to_mono_float(audio: tuple | None) -> np.ndarray | None:
    """Gradio Audio -> float32 mono waveform at SR."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2:
        sr, data = audio
    else:
        return None

    y = np.asarray(data, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Gradio sometimes gives int16
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.iinfo(data.dtype).max

    if sr != SR:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=SR)

    return y.astype(np.float32)


def _fix_length(y: np.ndarray, duration: float = DURATION) -> np.ndarray:
    target = int(SR * duration)
    if len(y) < target:
        return np.pad(y, (0, target - len(y)))
    # take last duration seconds (better for live stream)
    return y[-target:]


def waveform_to_mfcc(y: np.ndarray) -> np.ndarray:
    return extract_mfcc("", y=_fix_length(y))


@torch.no_grad()
def predict_prob(y: np.ndarray) -> tuple[float, float]:
    """Returns (angry_prob, inference_ms)."""
    feat = waveform_to_mfcc(y)
    x = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logit = MODEL(x).squeeze()
    prob = torch.sigmoid(logit).item()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return float(prob), ms


def format_result(
    prob: float,
    threshold: float,
    ms: float = 0.0,
    n_samples: int = 0,
    note: str = "",
) -> tuple[str, dict]:
    # 仅当概率严格超过阈值才判愤怒；相等/低于 → 非愤怒（常规默认）
    is_angry = prob > threshold
    label = "愤怒 ANGRY" if is_angry else "非愤怒 NON-ANGRY"
    color = "#c0392b" if is_angry else "#1e7a4a"
    bar = int(round(prob * 100))
    meta_bits = [f"愤怒概率&nbsp;<b style='color:#222;font-size:18px;'>{prob:.1%}</b>", f"阈值 {threshold:.2f}"]
    if ms > 0:
        meta_bits.append(f"推理 {ms:.1f} ms")
    if n_samples > 0:
        meta_bits.append(f"音频 {n_samples / SR:.2f}s")
    if note:
        meta_bits.append(note)
    meta = "&nbsp;·&nbsp;".join(meta_bits)

    html = f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;padding:8px 4px;">
      <div style="font-size:28px;font-weight:700;color:{color};letter-spacing:0.02em;">
        {label}
      </div>
      <div style="margin-top:14px;color:#555;font-size:14px;">{meta}</div>
      <div style="margin-top:12px;height:14px;background:#e8e6e1;border-radius:2px;overflow:hidden;">
        <div style="width:{bar}%;height:100%;background:{color};transition:width .2s;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:12px;color:#888;">
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
    </div>
    """
    # Label 优先显示非愤怒（常规状态）
    scores = {"非愤怒": float(1.0 - prob), "愤怒": float(prob)}
    return html, scores


def idle_result(threshold: float, note: str = "常规 / 待机") -> tuple[str, dict]:
    return format_result(0.0, float(threshold), note=note)


def analyze_audio(audio, threshold: float):
    y = _to_mono_float(audio)
    if y is None or len(y) < SR * 0.3:
        html, scores = idle_result(threshold, "请先录音或上传音频")
        return html, scores, None

    # energy gate: near silence → 非愤怒
    rms = float(np.sqrt(np.mean(y**2)))
    if rms < 0.008:
        html, scores = idle_result(threshold, "静音 / 音量过低 → 非愤怒")
        return html, scores, (SR, _fix_length(y))

    prob, ms = predict_prob(y)
    html, scores = format_result(prob, float(threshold), ms, len(_fix_length(y)))
    return html, scores, (SR, _fix_length(y))


def stream_analyze(audio, threshold: float, history: list | None):
    """Live mic stream: keep rolling 2s buffer and classify."""
    if history is None:
        history = []

    y = _to_mono_float(audio)
    if y is None or len(y) == 0:
        html, scores = idle_result(threshold, "等待麦克风 · 常规非愤怒")
        return html, scores, history

    history.append(y)
    buf = np.concatenate(history)
    # keep last 3s raw, classify last 2s
    max_keep = int(SR * 3.0)
    if len(buf) > max_keep:
        buf = buf[-max_keep:]
        history = [buf]

    if len(buf) < int(SR * 0.8):
        html, scores = idle_result(
            threshold, f"缓冲中 {len(buf)/SR:.1f}s / 2.0s · 常规非愤怒"
        )
        return html, scores, history

    rms = float(np.sqrt(np.mean(buf[-int(SR * DURATION) :] ** 2)))
    if rms < 0.008:
        html, scores = idle_result(threshold, "静音中 · 常规非愤怒")
        return html, scores, history

    prob, ms = predict_prob(buf)
    html, scores = format_result(prob, float(threshold), ms, int(SR * DURATION))
    return html, scores, history


CUSTOM_CSS = """
.gradio-container {
  max-width: 920px !important;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}
footer { display: none !important; }
#title-block {
  background: linear-gradient(160deg, #1a1f1c 0%, #2c3530 55%, #3d4a3f 100%);
  color: #f2f0ea;
  padding: 28px 28px 22px;
  border-radius: 4px;
  margin-bottom: 8px;
}
#title-block h1 {
  margin: 0 0 8px 0;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 0.04em;
}
#title-block p {
  margin: 0;
  opacity: 0.78;
  font-size: 14px;
  line-height: 1.5;
}
"""


def build_ui() -> gr.Blocks:
    meta_line = (
        f"设备 {META['gpu']} · 最优 epoch {META['best_epoch']} · "
        f"Val Acc {META['val_acc']:.1%} · Val F1 {META.get('val_f1', 0):.3f}"
    )

    with gr.Blocks(title="TinyCNN 愤怒检测", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(
            f"""
            <div id="title-block">
              <h1>TinyCNN · 实时愤怒检测</h1>
              <p>对着麦克风说话约 2 秒，模型输出愤怒概率。{meta_line}</p>
            </div>
            """
        )

        threshold = gr.Slider(
            0.1,
            0.9,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            label="判定阈值（默认 0.65；越高越不易判为愤怒）",
        )

        idle_html, _ = idle_result(DEFAULT_THRESHOLD, "常规状态 · 非愤怒")

        with gr.Tabs():
            with gr.Tab("录音 / 上传"):
                with gr.Row():
                    audio_in = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="录音或上传 wav",
                        format="wav",
                    )
                with gr.Row():
                    btn = gr.Button("开始检测", variant="primary")
                    clear_btn = gr.Button("清空")
                result_html = gr.HTML(value=idle_html)
                result_label = gr.Label(
                    value=DEFAULT_SCORES, label="类别概率", num_top_classes=2
                )
                preview = gr.Audio(label="实际送入模型的 2 秒片段", type="numpy")

                btn.click(
                    analyze_audio,
                    inputs=[audio_in, threshold],
                    outputs=[result_html, result_label, preview],
                )
                clear_btn.click(
                    lambda t: (None, *idle_result(t, "已清空 · 常规非愤怒"), None),
                    inputs=[threshold],
                    outputs=[audio_in, result_html, result_label, preview],
                )

            with gr.Tab("实时流式"):
                gr.Markdown(
                    "点击麦克风开始持续监听。系统用最近 **2 秒**滑动窗口实时推理。"
                    "待机 / 静音默认显示 **非愤怒**；仅当愤怒概率超过阈值才报警。"
                )
                stream_in = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="实时麦克风",
                )
                stream_html = gr.HTML(value=idle_html)
                stream_label = gr.Label(
                    value=DEFAULT_SCORES, label="实时概率", num_top_classes=2
                )
                stream_state = gr.State([])

                stream_in.stream(
                    stream_analyze,
                    inputs=[stream_in, threshold, stream_state],
                    outputs=[stream_html, stream_label, stream_state],
                    time_limit=None,
                    stream_every=0.5,
                )

        with gr.Accordion("使用说明", open=False):
            gr.Markdown(
                """
- **常规状态默认为非愤怒**：待机、静音、缓冲中、概率 ≤ 阈值 均判非愤怒。
- 默认阈值 **0.65**（仅 `概率 > 阈值` 才判愤怒）。
- 模型按 **2 秒 / 16 kHz / MFCC(40)** 训练，过短补零，过长取末尾 2 秒。
- 实时流式需浏览器麦克风权限。
- 权重路径：`checkpoints/best_model.pth`
                """
            )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True,
    )

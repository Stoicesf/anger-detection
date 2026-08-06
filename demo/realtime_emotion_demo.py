"""Realtime anger alarm demo — Phase 1 decision layer.

EMA + consecutive trigger + NORMAL/SUSPECT/ANGER/RECOVER state machine.
Default user-facing state is non-anger; alarm only in ANGER.

Run:
  conda activate pytorch12
  streamlit run demo/realtime_emotion_demo.py --server.port 8503
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(DEMO_DIR))

from common import MODELS_DIR, SR, TIME_FRAMES  # noqa: E402
from decision import AlarmState, AngerStateMachine, DecisionConfig, STATE_LABEL_ZH  # noqa: E402
from live_monitor import LiveAngerMonitor, audio_to_temp_wav  # noqa: E402
from prepare_data import log_mel_full  # noqa: E402

ANGRY_IDX = 2
LABEL_SHORT = ["中性", "高兴", "愤怒", "悲伤", "惊讶"]
STATE_COLOR = {
    "NORMAL": "#2bb673",
    "SUSPECT": "#e6a700",
    "ANGER": "#e53935",
    "RECOVER": "#5b8def",
}

CANDIDATES = [
    MODELS_DIR / "dscnn_casia_deploy.onnx",
    MODELS_DIR / "dscnn_casia.onnx",
    MODELS_DIR / "dscnn.onnx",
]


@st.cache_resource
def load_session(model_path: str):
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


@st.cache_resource
def get_monitor() -> LiveAngerMonitor:
    return LiveAngerMonitor()


def pick_model() -> Path:
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No ONNX model found.")


def center_crop(feat: np.ndarray) -> np.ndarray:
    t = feat.shape[0]
    if t >= TIME_FRAMES:
        start = (t - TIME_FRAMES) // 2
        return feat[start : start + TIME_FRAMES]
    pad_before = (TIME_FRAMES - t) // 2
    return np.pad(
        feat, ((pad_before, TIME_FRAMES - t - pad_before), (0, 0)), mode="edge"
    )


def predict_wav(sess, inp_name: str, wav_path: Path) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    feat = center_crop(log_mel_full(wav_path))
    x = feat.reshape(1, 1, TIME_FRAMES, 40).astype(np.float32)
    probs = sess.run(None, {inp_name: x})[0][0]
    return probs, time.perf_counter() - t0


def make_predict_fn(sess, inp_name: str):
    def _predict(audio: np.ndarray) -> tuple[np.ndarray, float]:
        return predict_wav(sess, inp_name, audio_to_temp_wav(audio, SR))

    return _predict


def record_seconds(seconds: float, sr: int = SR) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


st.set_page_config(page_title="愤怒事件检测 v1", page_icon="◈", layout="wide")
st.markdown(
    """
<style>
  .stApp {
    background:
      radial-gradient(900px 500px at 10% -10%, #1b2a3a 0%, transparent 55%),
      linear-gradient(160deg, #0f1720 0%, #152028 45%, #0c1218 100%) !important;
  }
  .block-container { max-width: 1100px; padding-top: 1.2rem; }
  h1, h2, h3, p, label { color: #e8f1ef !important; }
  .hero { font-size: 2rem; font-weight: 750; margin: 0; color: #eaf6f2; }
  .sub { color: #8fb0a8 !important; margin: 0.35rem 0 1.2rem; }
  .emo-card {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px; padding: 26px 28px;
    background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
  }
  .emo-name { font-size: 2.4rem; font-weight: 800; margin: 0.25rem 0; }
  .muted { color: #8aa39a !important; font-size: 0.92rem; }
  .pill {
    display:inline-block; padding:4px 12px; border-radius:999px;
    border:1px solid rgba(255,255,255,0.16); font-size:0.85rem; margin-right:8px;
  }
  .live-dot {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:8px; background:#2bb673;
    animation: pulse 1.4s infinite;
  }
  .live-dot.off { background:#666; animation:none; }
  .live-dot.hot { background:#e53935; }
  .live-dot.warn { background:#e6a700; }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(43,182,115,0.55); }
    70% { box-shadow: 0 0 0 12px rgba(43,182,115,0); }
    100% { box-shadow: 0 0 0 0 rgba(43,182,115,0); }
  }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="hero">愤怒事件检测 · Phase 1</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">EMA 平滑 · 连续触发 · NORMAL → SUSPECT → ANGER → RECOVER</p>',
    unsafe_allow_html=True,
)

try:
    model_path = pick_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

sess, inp_name = load_session(str(model_path))
monitor = get_monitor()
st.caption(f"模型：`{model_path.name}` · 分支 v1 / 决策层增强")

with st.sidebar:
    st.markdown("### 实时窗口")
    window_s = st.slider("分析窗口 (秒)", 0.8, 2.5, 1.0, 0.1)
    hop_s = st.slider("检测间隔 (秒)", 0.25, 1.0, 0.4, 0.05)
    st.markdown("### 决策层（Phase 1）")
    ema_alpha = st.slider("EMA α", 0.10, 0.60, 0.30, 0.05)
    anger_threshold = st.slider("触发阈值", 0.50, 0.90, 0.70, 0.05)
    clear_threshold = st.slider("解除阈值", 0.10, 0.60, 0.35, 0.05)
    required_hits = st.slider("连续命中次数", 1, 6, 3, 1)
    recover_hold_s = st.slider("愤怒后低分维持 (秒)", 1.0, 15.0, 3.0, 0.5)
    require_argmax = st.toggle("愤怒须为最高类", value=True)
    min_rms = st.slider("VAD 最小 RMS", 0.0, 0.05, 0.005, 0.001)
    st.markdown("### 单次检测")
    seconds = st.slider("单次录音 (秒)", 1.0, 5.0, 2.0, 0.5)
    st.caption("建议：窗口 1s · 间隔 0.4s · α=0.3 · 阈值 0.70 · 连续 3 次")

monitor.update_config(
    window_s=float(window_s),
    hop_s=float(hop_s),
    anger_threshold=float(anger_threshold),
    clear_threshold=float(clear_threshold),
    require_argmax=bool(require_argmax),
    min_rms=float(min_rms),
    ema_alpha=float(ema_alpha),
    required_hits=int(required_hits),
    recover_hold_s=float(recover_hold_s),
    sr=SR,
)

c1, c2, c3, c4 = st.columns(4)
start_live = c1.button("▶ 启动实时检测", type="primary")
stop_live = c2.button("■ 停止")
once = c3.button("单次录音")
reset = c4.button("重置状态")

if start_live:
    monitor.start(make_predict_fn(sess, inp_name))
    st.toast("实时检测已启动", icon="🎙️")
    time.sleep(0.15)
    st.rerun()

if stop_live:
    monitor.stop()
    st.toast("已停止", icon="⏹️")
    st.rerun()

if reset:
    monitor.reset_state()
    st.rerun()


@st.fragment(run_every=0.4)
def live_panel():
    snap = monitor.snapshot()
    color = STATE_COLOR.get(snap.state, "#2bb673")
    if snap.running:
        if snap.state == "ANGER":
            dot = "live-dot hot"
        elif snap.state == "SUSPECT":
            dot = "live-dot warn"
        else:
            dot = "live-dot"
        status = "LIVE"
    else:
        dot = "live-dot off"
        status = "IDLE"

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="emo-card">
              <div class="muted"><span class="{dot}"></span>{status}
              · {snap.state}
              · 帧 #{snap.frames}
              · RMS {snap.rms:.4f}</div>
              <div class="emo-name" style="color:{color}">{snap.state_zh}</div>
              <div class="muted">{snap.detail}</div>
              <div class="muted" style="margin-top:6px">
                平滑分 {snap.anger_p*100:.1f}%
                · 原始 {snap.raw_anger_p*100:.1f}%
                · 命中 {snap.hit_count}/{snap.required_hits}
                · {snap.lat_ms:.1f} ms
              </div>
              <div style="margin-top:10px">
                <span class="pill">EMA α={ema_alpha:.2f}</span>
                <span class="pill">阈值 {anger_threshold:.0%}</span>
                <span class="pill">连续×{required_hits}</span>
                <span class="pill">{window_s:.1f}s / {hop_s:.2f}s</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(
            min(max(snap.anger_p, 0.0), 1.0),
            text=f"EMA 愤怒分 {snap.anger_p*100:.1f}%",
        )
        if snap.error:
            st.warning(snap.error)
        if snap.probs is not None:
            with st.expander("五类原始概率"):
                st.bar_chart(
                    {LABEL_SHORT[i]: float(snap.probs[i]) for i in range(5)},
                    horizontal=True,
                )

    with right:
        st.markdown("#### 状态事件")
        if not snap.history:
            st.caption("启动实时检测后，状态迁移会显示在这里。")
        for h in snap.history[:16]:
            icon = {"ANGER": "🔴", "SUSPECT": "🟡", "RECOVER": "🔵"}.get(
                h.get("state", ""), "🟢"
            )
            st.write(
                f"{icon} `{h['t']}` **{h['label']}** "
                f"EMA {h['anger_p']*100:.0f}% "
                f"· {h.get('detail','')}"
            )


live_panel()

if once:
    if monitor.snapshot().running:
        st.warning("请先停止实时检测（避免抢麦）。")
    else:
        with st.spinner(f"录音 {seconds:.1f}s…"):
            try:
                audio = record_seconds(float(seconds))
            except Exception as e:
                st.error(f"麦克风失败：{e}")
                st.stop()
        rms = float(np.sqrt((audio**2).mean()))
        sm = AngerStateMachine(
            DecisionConfig(
                ema_alpha=float(ema_alpha),
                threshold=float(anger_threshold),
                clear_threshold=float(clear_threshold),
                required_hits=1,  # single-shot: one frame decision after EMA init
                require_argmax=bool(require_argmax),
            )
        )
        if rms < float(min_rms):
            st.info("音量过低 → 非愤怒")
        else:
            probs, lat = make_predict_fn(sess, inp_name)(audio)
            out = sm.update(
                float(probs[ANGRY_IDX]),
                top_idx=int(np.argmax(probs)),
                now=time.time(),
            )
            # For single clip, treat threshold crossing as alarm hint
            alarm = out.smooth_score > float(anger_threshold) and (
                (not require_argmax) or int(np.argmax(probs)) == ANGRY_IDX
            )
            label = "愤怒" if alarm else "非愤怒"
            if alarm:
                st.error(
                    f"**{label}** EMA {out.smooth_score*100:.1f}% "
                    f"(raw {out.raw_score*100:.1f}%) · {lat*1000:.1f}ms"
                )
            else:
                st.success(
                    f"**{label}** EMA {out.smooth_score*100:.1f}% "
                    f"(raw {out.raw_score*100:.1f}%) · {lat*1000:.1f}ms"
                )
            st.bar_chart(
                {LABEL_SHORT[i]: float(probs[i]) for i in range(5)}, horizontal=True
            )

uploaded = st.file_uploader("上传音频测试", type=["wav", "mp3", "flac", "ogg"])
if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".wav"
    tmp = Path(tempfile.gettempdir()) / f"emo_upload{suffix}"
    tmp.write_bytes(uploaded.read())
    probs, lat = predict_wav(sess, inp_name, tmp)
    raw = float(probs[ANGRY_IDX])
    alarm = raw > float(anger_threshold) and (
        (not require_argmax) or int(np.argmax(probs)) == ANGRY_IDX
    )
    if alarm:
        st.error(f"文件：**愤怒** raw {raw*100:.1f}% · {lat*1000:.1f}ms")
    else:
        st.success(f"文件：**非愤怒** raw {raw*100:.1f}% · {lat*1000:.1f}ms")

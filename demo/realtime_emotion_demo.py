"""Realtime anger alarm demo (fine-tuned DS-CNN + ONNX).

Default state = 非愤怒. Supports continuous live mic monitoring.

Run:
  conda activate pytorch12
  cd "E:\\桌面\\Anger detection"
  streamlit run demo/realtime_emotion_demo.py --server.port 8503
"""
from __future__ import annotations

import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(DEMO_DIR))

from common import MODELS_DIR, SR, TIME_FRAMES  # noqa: E402
from live_monitor import LiveAngerMonitor, audio_to_temp_wav  # noqa: E402
from prepare_data import log_mel_full  # noqa: E402

ANGRY_IDX = 2
LABEL_SHORT = ["中性", "高兴", "愤怒", "悲伤", "惊讶"]

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
    raise FileNotFoundError("No ONNX model found. Run scripts/finetune_casia.py first.")


def center_crop(feat: np.ndarray) -> np.ndarray:
    t = feat.shape[0]
    if t >= TIME_FRAMES:
        start = (t - TIME_FRAMES) // 2
        return feat[start : start + TIME_FRAMES]
    pad_before = (TIME_FRAMES - t) // 2
    return np.pad(
        feat,
        ((pad_before, TIME_FRAMES - t - pad_before), (0, 0)),
        mode="edge",
    )


def predict_wav(sess, inp_name: str, wav_path: Path) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    feat = center_crop(log_mel_full(wav_path))
    x = feat.reshape(1, 1, TIME_FRAMES, 40).astype(np.float32)
    probs = sess.run(None, {inp_name: x})[0][0]
    return probs, time.perf_counter() - t0


def make_predict_fn(sess, inp_name: str):
    def _predict(audio: np.ndarray) -> tuple[np.ndarray, float]:
        wav = audio_to_temp_wav(audio, SR)
        return predict_wav(sess, inp_name, wav)

    return _predict


def decide_state(
    probs: np.ndarray,
    *,
    anger_threshold: float,
    require_argmax: bool,
    prev_angry: bool,
    clear_threshold: float,
) -> tuple[bool, float, str]:
    anger_p = float(probs[ANGRY_IDX])
    top = int(np.argmax(probs))
    if prev_angry:
        if anger_p >= clear_threshold:
            return True, anger_p, "保持愤怒（未回落）"
        return False, anger_p, "愤怒已解除"
    triggered = anger_p >= anger_threshold
    if require_argmax:
        triggered = triggered and top == ANGRY_IDX
    if triggered:
        return True, anger_p, "检测到愤怒"
    return False, anger_p, "常规（非愤怒）"


def record_seconds(seconds: float, sr: int = SR) -> np.ndarray:
    import sounddevice as sd

    frames = int(seconds * sr)
    audio = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


st.set_page_config(page_title="愤怒状态监测", page_icon="◈", layout="wide")
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
  .emo-name { font-size: 2.6rem; font-weight: 800; margin: 0.25rem 0; }
  .muted { color: #8aa39a !important; font-size: 0.92rem; }
  .pill {
    display:inline-block; padding:4px 12px; border-radius:999px;
    border:1px solid rgba(255,255,255,0.16); font-size:0.85rem; margin-right:8px;
  }
  .live-dot {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:8px; background:#2bb673;
    box-shadow: 0 0 0 0 rgba(43,182,115,0.7);
    animation: pulse 1.4s infinite;
  }
  .live-dot.off { background:#666; box-shadow:none; animation:none; }
  .live-dot.hot { background:#e53935; box-shadow: 0 0 0 0 rgba(229,57,53,0.7); }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(43,182,115,0.55); }
    70% { box-shadow: 0 0 0 12px rgba(43,182,115,0); }
    100% { box-shadow: 0 0 0 0 rgba(43,182,115,0); }
  }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="hero">愤怒状态监测</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">默认「非愤怒」· 实时麦克风持续检测 · 仅愤怒过阈值才切换</p>',
    unsafe_allow_html=True,
)

try:
    model_path = pick_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

sess, inp_name = load_session(str(model_path))
monitor = get_monitor()
st.caption(f"模型：`{model_path.name}`")

with st.sidebar:
    st.markdown("### 实时检测")
    window_s = st.slider("分析窗口 (秒)", 1.0, 3.0, 1.5, 0.5)
    hop_s = st.slider("检测间隔 (秒)", 0.3, 2.0, 0.5, 0.1)
    anger_threshold = st.slider("触发愤怒阈值", 0.30, 0.95, 0.55, 0.05)
    clear_threshold = st.slider("解除愤怒阈值", 0.10, 0.80, 0.35, 0.05)
    require_argmax = st.toggle("愤怒须为最高类", value=True)
    min_rms = st.slider("最小音量 (RMS)", 0.0, 0.05, 0.005, 0.001)
    st.markdown("### 单次检测")
    seconds = st.slider("单次录音时长 (秒)", 1.0, 5.0, 2.0, 0.5)
    st.markdown("### 逻辑")
    st.write(
        "- 点 **启动实时检测** 后持续听麦\n"
        "- 常规 = 非愤怒；仅愤怒过阈值报警\n"
        "- 静音自动回到非愤怒"
    )

monitor.update_config(
    window_s=float(window_s),
    hop_s=float(hop_s),
    anger_threshold=float(anger_threshold),
    clear_threshold=float(clear_threshold),
    require_argmax=bool(require_argmax),
    min_rms=float(min_rms),
    sr=SR,
)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
start_live = c1.button("▶ 启动实时检测", type="primary")
stop_live = c2.button("■ 停止实时检测")
once = c3.button("单次录音检测")
reset = c4.button("重置为非愤怒")

if start_live:
    monitor.start(make_predict_fn(sess, inp_name))
    st.toast("实时检测已启动", icon="🎙️")
    time.sleep(0.2)
    st.rerun()

if stop_live:
    monitor.stop()
    st.toast("已停止", icon="⏹️")
    st.rerun()

if reset:
    monitor.stop()
    with monitor._lock:
        monitor._snap.is_angry = False
        monitor._snap.anger_p = 0.0
        monitor._snap.detail = "已手动重置"
        monitor._snap.probs = None
    st.rerun()


@st.fragment(run_every=0.5)
def live_panel():
    snap = monitor.snapshot()
    is_angry = bool(snap.is_angry)
    state_name = "愤怒" if is_angry else "非愤怒"
    state_color = "#e53935" if is_angry else "#2bb673"
    dot_cls = "live-dot hot" if is_angry else ("live-dot" if snap.running else "live-dot off")
    status = "LIVE" if snap.running else "IDLE"

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="emo-card">
              <div class="muted"><span class="{dot_cls}"></span>{status}
              · 帧 #{snap.frames}
              · RMS {snap.rms:.4f}</div>
              <div class="emo-name" style="color:{state_color}">{state_name}</div>
              <div class="muted">{snap.detail}
              · 愤怒分 {snap.anger_p*100:.1f}%
              · {snap.lat_ms:.1f} ms</div>
              <div style="margin-top:10px">
                <span class="pill">默认：非愤怒</span>
                <span class="pill">阈值：{anger_threshold:.0%}</span>
                <span class="pill">窗口：{window_s:.1f}s / {hop_s:.1f}s</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(
            min(max(snap.anger_p, 0.0), 1.0),
            text=f"愤怒置信度 {snap.anger_p*100:.1f}%",
        )
        if snap.error:
            st.warning(snap.error)
        if snap.probs is not None:
            with st.expander("五类原始概率（调试）"):
                st.bar_chart(
                    {LABEL_SHORT[i]: float(snap.probs[i]) for i in range(5)},
                    horizontal=True,
                )

    with right:
        st.markdown("#### 事件日志")
        hist = snap.history
        if not hist:
            st.caption("尚无事件。启动实时检测后，状态变化会显示在这里。")
        for h in hist[:14]:
            mark = "🔴" if h["angry"] else "🟢"
            st.write(
                f"{mark} `{h['t']}`  **{h['label']}**  "
                f"愤怒分 {h['anger_p']*100:.0f}%  ·  {h['detail']}  ·  {h['lat_ms']:.0f}ms"
            )


live_panel()

if once:
    if monitor.snapshot().running:
        st.warning("请先停止实时检测，再使用单次录音（避免抢占麦克风）。")
    else:
        with st.spinner(f"录音 {seconds:.1f}s…"):
            try:
                audio = record_seconds(float(seconds))
            except Exception as e:
                st.error(f"麦克风失败：{e}")
                st.stop()
        rms = float(np.sqrt((audio**2).mean()))
        if rms < float(min_rms):
            st.info("音量过低，判定为非愤怒")
        else:
            probs, lat = make_predict_fn(sess, inp_name)(audio)
            is_angry_now, anger_p, detail = decide_state(
                probs,
                anger_threshold=float(anger_threshold),
                require_argmax=bool(require_argmax),
                prev_angry=False,
                clear_threshold=float(clear_threshold),
            )
            if is_angry_now:
                st.error(f"**愤怒** {anger_p*100:.1f}% · {detail} · {lat*1000:.1f}ms")
            else:
                st.success(f"**非愤怒**（愤怒分 {anger_p*100:.1f}%）· {detail} · {lat*1000:.1f}ms")
            st.bar_chart(
                {LABEL_SHORT[i]: float(probs[i]) for i in range(5)}, horizontal=True
            )

uploaded = st.file_uploader("或上传音频测试", type=["wav", "mp3", "flac", "ogg"])
if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".wav"
    tmp = Path(tempfile.gettempdir()) / f"emo_upload{suffix}"
    tmp.write_bytes(uploaded.read())
    probs, lat = predict_wav(sess, inp_name, tmp)
    is_angry_now, anger_p, detail = decide_state(
        probs,
        anger_threshold=float(anger_threshold),
        require_argmax=bool(require_argmax),
        prev_angry=False,
        clear_threshold=float(clear_threshold),
    )
    if is_angry_now:
        st.error(f"文件：**愤怒**（{anger_p*100:.1f}%）· {detail}")
    else:
        st.success(f"文件：**非愤怒**（愤怒分 {anger_p*100:.1f}%）· {detail}")

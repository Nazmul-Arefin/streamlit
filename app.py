# app.py — VisionSpeak (PaddleOCR 2.6.1 API)
# Flow: Upload image → OCR (use_angle_cls=True) → show text + annotated image → TTS (pyttsx3)

import os
import io
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import cv2

from paddleocr import PaddleOCR
import pyttsx3

# ---------------- Page & styles ----------------
st.set_page_config(page_title="VisionSpeak", page_icon="🗣️", layout="wide")
st.markdown("""
<style>
    /* ===== App-wide look ===== */
    .block-container { max-width: 1180px; position: relative; z-index: 2; }
    .stApp { background: transparent; }

    /* Animated gradient backdrop */
    #vs-anim-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
    background: linear-gradient(120deg, #0f1226, #0b213b, #091a2a);
    }

    /* Slow gradient shimmer */
    #vs-anim-bg::before {
    content: "";
    position: absolute; inset: -20% -20%;
    background: radial-gradient(60% 60% at 20% 30%, rgba(0,170,255,0.12) 0%, rgba(0,0,0,0) 60%),
                radial-gradient(55% 55% at 80% 70%, rgba(170,0,255,0.12) 0%, rgba(0,0,0,0) 60%),
                radial-gradient(40% 40% at 40% 80%, rgba(0,255,170,0.10) 0%, rgba(0,0,0,0) 60%);
    filter: blur(40px);
    animation: bg-pan 24s ease-in-out infinite alternate;
    }

    /* Floating blobs */
    .vs-blob {
    position: absolute;
    width: 42vmin; height: 42vmin;
    border-radius: 50%;
    opacity: 0.10;
    filter: blur(24px);
    background: radial-gradient(circle at 30% 30%, rgba(0,180,255,0.9), rgba(0,180,255,0) 60%);
    animation: blob-float 26s ease-in-out infinite;
    }
    .vs-blob.b2 {
    left: 65%; top: 10%;
    background: radial-gradient(circle at 70% 70%, rgba(190,0,255,0.9), rgba(190,0,255,0) 60%);
    animation-duration: 30s;
    }
    .vs-blob.b3 {
    left: 15%; top: 70%;
    background: radial-gradient(circle at 50% 40%, rgba(0,255,190,0.9), rgba(0,255,190,0) 60%);
    animation-duration: 34s;
    }

    /* Cards & text */
    .vs-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px; padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
    }
    .vs-text {
    font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace;
    white-space: pre-wrap; line-height: 1.6;
    }

    /* Buttons */
    .stButton>button {
    border-radius: 999px; padding: 0.6rem 1rem;
    border: 1px solid rgba(255,255,255,0.20);
    background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
    transition: transform .15s ease, box-shadow .2s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,0,0,.25); }

    /* Animated gradient title */
    .vs-title {
    font-size: clamp(28px, 5vw, 44px);
    font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #8bd3ff, #c9a2ff, #91ffd8, #8bd3ff);
    background-size: 300% 100%;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: hue-slide 10s linear infinite;
    }

    /* Keyframes */
    @keyframes bg-pan {
    0% { transform: translate3d(-6%, -4%, 0) scale(1.05); }
    100% { transform: translate3d(6%, 4%, 0) scale(1.05); }
    }
    @keyframes blob-float {
    0%   { transform: translate3d(-5%, 0, 0) scale(1); }
    50%  { transform: translate3d(5%, -3%, 0) scale(1.08); }
    100% { transform: translate3d(-5%, 0, 0) scale(1); }
    }
    @keyframes hue-slide {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
    }

    /* ===== Sidebar glass + accents ===== */
    [data-testid="stSidebar"] {
    background: transparent;
    }

    [data-testid="stSidebar"] > div:first-child {
    /* glass card */
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    margin: 12px;
    padding: 12px 12px 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
    }

    /* thin animated accent bar on the left */
    [data-testid="stSidebar"] > div:first-child::before {
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, #8bd3ff, #c9a2ff, #91ffd8, #8bd3ff);
    background-size: 100% 300%;
    animation: hue-slide 10s linear infinite;
    opacity: .9;
    }

    /* sidebar headings */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
    letter-spacing: -0.02em;
    margin-top: 8px;
    }

    /* tidy widget cards */
    [data-testid="stSidebar"] .stSlider, 
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stTextInput, 
    [data-testid="stSidebar"] .stCheckbox {
    padding: 6px 8px;
    border-radius: 12px;
    transition: background .15s ease;
    }
    [data-testid="stSidebar"] .stSlider:hover, 
    [data-testid="stSidebar"] .stSelectbox:hover, 
    [data-testid="stSidebar"] .stTextInput:hover, 
    [data-testid="stSidebar"] .stCheckbox:hover {
    background: rgba(255,255,255,0.06);
    }

    /* prettier slider */
    [data-testid="stSlider"] [role="slider"] {
    box-shadow: 0 0 0 6px rgba(139,211,255,0.25);
    border-radius: 50%;
    }
    [data-testid="stSlider"] .st-emotion-cache-14f2vti, 
    [data-testid="stSlider"] .st-emotion-cache-1u8cb6j {
    /* track (Streamlit classnames vary across versions; this hits most) */
    height: 4px !important;
    background: linear-gradient(90deg, #8bd3ff, #c9a2ff, #91ffd8);
    }

    /* uploader hover */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border-radius: 12px;
    border: 1px dashed rgba(255,255,255,0.25);
    padding: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    background: rgba(255,255,255,0.05);
    }

    /* subtle footer text */
    .vs-side-foot {
    font-size: 12px; opacity: .75; text-align: center; margin-top: 10px;
    }
       
</style>
""", unsafe_allow_html=True)

# Animated background layer (behind all content)
st.markdown("""
<div id="vs-anim-bg">
  <div class="vs-blob" style="left:-10%; top:10%;"></div>
  <div class="vs-blob b2"></div>
  <div class="vs-blob b3"></div>
</div>
""", unsafe_allow_html=True)


# ---------------- Simple visualizer (replacement for draw_ocr) ----------------
def draw_ocr_simple(np_img: np.ndarray, boxes: List[List[List[float]]], txts: Optional[List[str]] = None) -> Image.Image:
    """Draw quadrilateral boxes and optional text labels on an image."""
    img = Image.fromarray(np_img.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        pts = [(int(x), int(y)) for x, y in box]
        draw.line(pts + [pts[0]], width=2, fill=(0, 255, 0))  # outline
        if txts and i < len(txts) and txts[i]:
            label = str(txts[i])
            x, y = pts[0]
            y = max(0, y - 16)
            w = 8 * len(label) + 6
            h = 16
            draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
            if font:
                draw.text((x + 3, y), label, fill=(255, 255, 255), font=font)
            else:
                draw.text((x + 3, y), label, fill=(255, 255, 255))
    return img

# ---------------- Normalize PaddleOCR outputs ----------------
def normalize_paddle_result(result: Any) -> Tuple[List[List[List[float]]], List[str], List[float]]:
    """Normalize PaddleOCR outputs to (boxes, texts, scores)."""
    boxes: List[List[List[float]]] = []
    txts: List[str] = []
    scores: List[float] = []

    if result is None:
        return boxes, txts, scores

    pages = result if isinstance(result, list) else [result]

    for page in pages:
        if not page:
            continue
        if isinstance(page, list) and page and isinstance(page[0], (list, tuple)) and len(page[0]) >= 2:
            for det in page:
                try:
                    box = det[0]
                    info = det[1]
                    text = info[0] if isinstance(info, (list, tuple)) else str(info)
                    score = info[1] if isinstance(info, (list, tuple)) and len(info) > 1 else 1.0
                    boxes.append(box); txts.append(text); scores.append(float(score))
                except Exception:
                    continue
            continue
    return boxes, txts, scores

# ---------------- Light preprocessing (silent fallback) ----------------
def preprocess_for_ocr(img_pil: Image.Image) -> Image.Image:
    """Upscale + adaptive threshold to help on low-contrast/small images."""
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]
    if min(h, w) < 720:
        scale = 720.0 / max(1, min(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 25, 10)
    cleaned = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(cleaned)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin:6px 2px 6px 2px;">
          <div style="width:30px;height:30px;border-radius:8px;
                      background:linear-gradient(135deg,#8bd3ff,#c9a2ff,#91ffd8);"></div>
          <div style="font-weight:800;letter-spacing:-0.02em;">VisionSpeak</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.header("⚙️ Settings")
    lang = st.selectbox("OCR language", ["en", "latin", "ch", "japan", "korean", "bangla"], index=0)
    use_gpu = st.checkbox("Use GPU (requires PaddlePaddle GPU build)", value=False)

    st.markdown("---")
    st.header("🔊 Speech")
    tts_rate = st.slider("Speech rate", 120, 220, 170, 5)
    tts_volume = st.slider("Volume", 0.2, 1.0, 1.0, 0.05)

    st.markdown('<div class="vs-side-foot">© VisionSpeak • Built with Streamlit</div>', unsafe_allow_html=True)


# ---------------- Cache OCR (2.6.1-style constructor) ----------------
@st.cache_resource(show_spinner=False)
def load_ocr_2x(lang: str, use_gpu: bool):
    # Same API you used in Flask
    try:
        return PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
    except Exception:
        # Fallback to CPU if GPU wheel isn't installed
        return PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False,
            show_log=False
        )

def run_ocr(img: Image.Image, lang: str, use_gpu: bool) -> Tuple[str, Optional[Image.Image]]:
    ocr = load_ocr_2x(lang, use_gpu)

    def _infer(arr: np.ndarray):
        # 2.x API supports .ocr(arr, cls=True)
        try:
            return ocr.ocr(arr, cls=True)
        except Exception:
            return None

    # Try original
    np_img = np.array(img.convert("RGB"))
    raw = _infer(np_img)

    # If empty, try cleaned and 4 rotations silently
    def is_empty(res):
        return (res is None) or (isinstance(res, list) and all(len(p) == 0 for p in res if isinstance(p, list)))

    if is_empty(raw):
        variants = [img, preprocess_for_ocr(img)]
        for base in variants:
            for angle in [0, 90, 180, 270]:
                test_img = base if angle == 0 else base.rotate(angle, expand=True)
                raw2 = _infer(np.array(test_img.convert("RGB")))
                if not is_empty(raw2):
                    raw = raw2
                    np_img = np.array(test_img.convert("RGB"))
                    break
            if not is_empty(raw):
                break

    boxes, txts, scores = normalize_paddle_result(raw)
    text = "\n".join(txts).strip()
    annotated = draw_ocr_simple(np_img, boxes, txts) if boxes else None
    return text, annotated

def synth_tts_pyttsx3(text: str, rate: int, volume: float) -> bytes:
    eng = pyttsx3.init()
    eng.setProperty("rate", rate)
    eng.setProperty("volume", volume)
    try:
        voices = eng.getProperty("voices")
        if voices:
            eng.setProperty("voice", voices[0].id)
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        eng.save_to_file(text, wav_path)
        eng.runAndWait()
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

# ---------------- Header ----------------
st.markdown('<div class="vs-title">VisionSpeak</div>', unsafe_allow_html=True)
st.subheader("Upload → OCR → TTS")


# ---------------- Upload & actions ----------------
st.markdown('<div class="vs-card">', unsafe_allow_html=True)
file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
cA, cB = st.columns(2)
with cA:
    run_btn = st.button("🔎 Extract Text", use_container_width=True)
with cB:
    tts_btn = st.button("🗣️ Convert Last Text to Speech", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- State ----------------
st.session_state.setdefault("text", "")
st.session_state.setdefault("img_src", None)
st.session_state.setdefault("img_ann", None)
st.session_state.setdefault("audio", b"")

# ---------------- Actions ----------------
if run_btn:
    if not file:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Running PaddleOCR…"):
            img = Image.open(file).convert("RGB")
            text, ann = run_ocr(img, lang=lang, use_gpu=use_gpu)
            st.session_state["text"] = text
            st.session_state["img_src"] = img
            st.session_state["img_ann"] = ann
            st.session_state["audio"] = b""

if tts_btn:
    if not st.session_state["text"].strip():
        st.warning("No text to speak; run OCR first.")
    else:
        with st.spinner("Synthesizing speech…"):
            try:
                st.session_state["audio"] = synth_tts_pyttsx3(
                    st.session_state["text"], tts_rate, tts_volume
                )
            except Exception as e:
                st.error(f"TTS failed: {e}")

# ---------------- Results ----------------
if st.session_state["text"] or st.session_state["img_src"]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🖼️ Image")
        st.markdown('<div class="vs-card">', unsafe_allow_html=True)
        if st.session_state["img_ann"] is not None:
            st.image(st.session_state["img_ann"], caption="Detected text (annotated)", use_container_width=True)
            with st.expander("Show original"):
                st.image(st.session_state["img_src"], use_container_width=True)
        elif st.session_state["img_src"] is not None:
            st.image(st.session_state["img_src"], caption="Uploaded image", use_container_width=True)
        else:
            st.info("No image yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### 📝 Extracted Text")
        st.markdown('<div class="vs-card">', unsafe_allow_html=True)
        if st.session_state["text"].strip():
            st.markdown(f'<div class="vs-text">{st.session_state["text"]}</div>', unsafe_allow_html=True)
            st.download_button(
                "Download text",
                data=st.session_state["text"].encode("utf-8"),
                file_name=f"visionspeak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Run OCR to see text here.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔊 Speech")
    st.markdown('<div class="vs-card">', unsafe_allow_html=True)
    if st.session_state["audio"]:
        st.audio(st.session_state["audio"], format="audio/wav")
        st.download_button(
            "Download audio (WAV)",
            data=st.session_state["audio"],
            file_name=f"visionspeak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
            mime="audio/wav",
            use_container_width=True
        )
    else:
        st.info("Click **Convert Last Text to Speech** to generate audio.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.caption("Upload an image to get started.")

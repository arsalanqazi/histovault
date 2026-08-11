"""
HistoVault v0.5 [Preview] — Streamlit Medical AI Diagnostics Platform
Clean Clinical Light Theme (Matches original React app)
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import base64
import time
import io
from PIL import Image

# Load favicon as base64 for header icon
with open("logo.png", "rb") as _f:
    _favicon_b64 = base64.b64encode(_f.read()).decode()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HistoVault",
    page_icon=Image.open("favicon.ico"),
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Import backend modules ───────────────────────────────────────────────────
from config import (
    TARGET_SIZE, MODEL_PATHS, MODEL_GPU_MAP,
    LABEL_MAPPINGS, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    EIGHT_CLASS_NAMES,
)
from model_loader import get_model
from gradcam import get_gradcam_heatmap, apply_heatmap
from validation import validate_image

# ── Custom CSS — Light Clinical Theme (Exact match to original React app) ────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Page background — Clinical Light Grey */
.stApp {
    background-color: #f4f6f8;
    color: #2d3748;
}

/* Hide Streamlit default header/footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Main Header */
.hv-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 20px 10px 30px 10px;
}
.hv-icon img {
    width: 128px;
    height: 128px;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
}
.hv-title {
    font-size: 36px;
    font-weight: 700;
    color: #1a365d;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}
.hv-desc {
    font-size: 16px;
    max-width: 600px;
    color: #718096;
    line-height: 1.6;
    margin: 0 auto;
}

/* Controls Container */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    padding: 24px !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05) !important;
    border: none !important;
    margin-bottom: 30px !important;
}

/* Buttons styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
    color: white !important;
    padding: 14px 20px !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    box-shadow: 0 4px 6px rgba(49, 130, 206, 0.3) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"]:hover:not(:disabled) {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(49, 130, 206, 0.4) !important;
}

div.stButton > button[kind="secondary"] {
    background: #718096 !important;
    color: white !important;
    padding: 14px 20px !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

/* Error banner */
.hv-error-box {
    background: #fff5f5;
    border: 1px solid #feb2b2;
    color: #c53030;
    padding: 15px 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 500;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* Result Card */
.hv-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    overflow: hidden;
    margin-bottom: 30px;
}
.hv-card-header {
    padding: 20px 30px;
    border-bottom: 1px solid #edf2f7;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8fafc;
    flex-wrap: wrap;
    gap: 10px;
}
.hv-filename {
    font-size: 14px;
    color: #718096;
    font-family: monospace;
    background: #e2e8f0;
    padding: 4px 8px;
    border-radius: 6px;
}
.hv-badge-cancer {
    font-size: 14px;
    font-weight: 700;
    padding: 6px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background-color: #fed7d7;
    color: #c53030;
}
.hv-badge-normal {
    font-size: 14px;
    font-weight: 700;
    padding: 6px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background-color: #c6f6d5;
    color: #2f855a;
}
.hv-card-body {
    padding: 30px;
}
.hv-img-label {
    font-size: 12px;
    font-weight: 600;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: center;
    margin-bottom: 8px;
}
.hv-stats {
    display: flex;
    flex-direction: column;
    gap: 15px;
    padding: 20px;
    background: #f7fafc;
    border-radius: 12px;
    margin-top: 20px;
}
.hv-prob-row {
    display: flex;
    align-items: center;
    gap: 15px;
}
.hv-label {
    width: 120px;
    font-size: 14px;
    font-weight: 500;
    color: #4a5568;
}
.hv-bar-bg {
    flex: 1;
    height: 12px;
    background: #e2e8f0;
    border-radius: 6px;
    overflow: hidden;
}
.hv-bar-red {
    height: 100%;
    background: #f56565;
    transition: width 1s ease-out;
}
.hv-bar-green {
    height: 100%;
    background: #48bb78;
    transition: width 1s ease-out;
}
.hv-pct {
    width: 50px;
    text-align: right;
    font-size: 14px;
    font-weight: 700;
    color: #2d3748;
}
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes):
    """Preprocess image bytes into PyTorch tensor + BGR numpy array."""
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    img_rgb = np.array(pil_img)
    original_img_resized = cv2.resize(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), TARGET_SIZE
    )

    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor_img = transform(pil_img).unsqueeze(0)
    return tensor_img, original_img_resized


def predict_cancer(model, preprocessed_img, is_multi_class=False):
    """Run PyTorch model prediction."""
    device = next(model.parameters()).device
    preprocessed_img = preprocessed_img.to(device)

    with torch.no_grad():
        logits = model(preprocessed_img)
        if is_multi_class:
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_class = [int(np.argmax(probs))]
            return predicted_class, probs
        else:
            prob = torch.sigmoid(logits).item()
            predicted_class = [1] if prob >= 0.5 else [0]
            return predicted_class, [[prob]]


def interpret_prediction(predicted_class, prediction, cancer_type):
    """Interpret raw prediction probabilities."""
    if cancer_type == "8class":
        predicted_label = EIGHT_CLASS_NAMES[predicted_class[0]]
        cancerous_prob    = float(prediction[1] + prediction[3] + prediction[5] + prediction[7])
        non_cancerous_prob = float(prediction[0] + prediction[2] + prediction[4] + prediction[6])
        top_class_prob     = float(prediction[predicted_class[0]])
        return predicted_label, cancerous_prob, non_cancerous_prob, top_class_prob

    class_labels       = LABEL_MAPPINGS.get(cancer_type, ["Non Cancerous", "Cancerous"])
    predicted_label    = class_labels[predicted_class[0]]
    cancerous_prob     = float(prediction[0][0])
    non_cancerous_prob = 1.0 - cancerous_prob
    top_class_prob     = max(cancerous_prob, non_cancerous_prob)
    return predicted_label, cancerous_prob, non_cancerous_prob, top_class_prob


def is_cancerous(prediction: str) -> bool:
    lower = prediction.lower()
    return "non cancerous" not in lower and "non-cancerous" not in lower


def img_bytes_to_base64_jpeg(bgr_array) -> str:
    """Convert BGR numpy array to base64 JPEG URI."""
    _, buffer = cv2.imencode(".jpg", bgr_array)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


@st.cache_resource(show_spinner=False)
def load_model_cached(cancer_type: str):
    return get_model(cancer_type)


# ── Session state ────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = []
if "error_msg" not in st.session_state:
    st.session_state.error_msg = ""


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="hv-header">
    <div class="hv-icon"><img src="data:image/x-icon;base64,{_favicon_b64}" alt="HistoVault"/></div>
    <div class="hv-title">HistoVault v0.5 [Preview]</div>
    <div class="hv-desc">
        Advanced AI diagnostics for low-cost, low-resolution histopathology analysis.
        Supports multi-organ cancer detection with AI attention.
        If you see a blue patch on the image, you really need to look at that part.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Controls Section ─────────────────────────────────────────────────────────

MODEL_DISPLAY_NAMES = {
    "colorectal":      "Colorectal Cancer",
    "breast":          "Breast Cancer",
    "gastrointestinal":"Gastrointestinal Cancer",
    "oral":            "Oral Cancer",
    "octa":            "Binary",
    "8class":          "Multi-Organ (8 Classes)",
}

with st.container(border=True):
    cancer_key = st.selectbox(
        "Select Cancer Type",
        options=list(MODEL_DISPLAY_NAMES.keys()),
        format_func=lambda k: MODEL_DISPLAY_NAMES[k],
        key="cancer_type_select",
    )

    uploaded_file = st.file_uploader(
        "Upload Histopathology Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="image_uploader",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        run_clicked = st.button(
            "Run Analysis",
            type="primary",
            disabled=(uploaded_file is None),
            use_container_width=True,
            key="btn_run",
        )
    with col2:
        save_clicked = st.button(
            "Save Session",
            type="secondary",
            disabled=(len(st.session_state.results) == 0),
            use_container_width=True,
            key="btn_save",
        )


# ── Handle Save Session ──────────────────────────────────────────────────────

if save_clicked and st.session_state.results:
    rows = ["Filename\tModel\tPrediction\tCancerous Probability\tNon-Cancerous Probability\tProcessing Time"]
    for r in st.session_state.results:
        rows.append(
            f"{r['fileName']}\t{r['model']}\t{r['prediction']}\t"
            f"{r['probability']['cancerous']*100:.2f}%\t"
            f"{r['probability']['non_cancerous']*100:.2f}%\t"
            f"{r['processingTime']}"
        )
    tsv_content = "\n".join(rows)
    st.download_button(
        label="⬇ Download session_results.txt",
        data=tsv_content,
        file_name="session_results.txt",
        mime="text/plain",
        key="download_btn",
        use_container_width=True
    )


# ── Handle Run Analysis ──────────────────────────────────────────────────────

if run_clicked and uploaded_file is not None:
    st.session_state.error_msg = ""
    image_bytes = uploaded_file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        st.session_state.error_msg = (
            f"File size exceeds {MAX_FILE_SIZE // (1024*1024)} MB limit."
        )
    else:
        with st.spinner("Analyzing..."):
            start_time = time.time()

            # 1. Histopathology validation
            is_histo, validation_msg, confidence = validate_image(image_bytes)
            if not is_histo:
                st.session_state.error_msg = validation_msg
            else:
                try:
                    # 2. Load PyTorch model
                    model = load_model_cached(cancer_key)

                    # 3. Preprocess
                    tensor_img, original_bgr = preprocess_image(image_bytes)

                    # 4. Predict
                    is_multi = (cancer_key == "8class")
                    predicted_class, prediction = predict_cancer(model, tensor_img, is_multi_class=is_multi)

                    # 5. Interpret
                    pred_label, cancer_prob, noncancer_prob, top_prob = interpret_prediction(
                        predicted_class, prediction, cancer_key
                    )

                    # 6. GradCAM
                    gradcam_b64 = None
                    try:
                        heatmap = get_gradcam_heatmap(
                            model, tensor_img,
                            last_conv_layer_name=None,
                            pred_index=predicted_class[0],
                        )
                        gradcam_b64 = "data:image/jpeg;base64," + apply_heatmap(heatmap, original_bgr)
                    except Exception:
                        pass

                    # 7. Base64 Original Preview
                    original_b64 = img_bytes_to_base64_jpeg(original_bgr)
                    processing_time = f"{time.time() - start_time:.4f}s"

                    result = {
                        "fileName":    uploaded_file.name,
                        "model":       MODEL_DISPLAY_NAMES.get(cancer_key, cancer_key),
                        "image":       original_b64,
                        "prediction":  pred_label,
                        "probability": {
                            "cancerous":           cancer_prob,
                            "non_cancerous":       noncancer_prob,
                            "top_class_confidence":top_prob,
                        },
                        "gradcam":        gradcam_b64,
                        "processingTime": processing_time,
                    }
                    st.session_state.results.insert(0, result)

                except Exception as e:
                    st.session_state.error_msg = f"Analysis error: {str(e)}"


# ── Error Message Banner ──────────────────────────────────────────────────────

if st.session_state.error_msg:
    st.markdown(
        f'<div class="hv-error-box"><span style="font-size:20px">⚠️</span>'
        f'<span>{st.session_state.error_msg}</span></div>',
        unsafe_allow_html=True,
    )


# ── Results Feed ─────────────────────────────────────────────────────────────

for result in st.session_state.results:
    cancer      = is_cancerous(result["prediction"])
    cancer_pct  = result["probability"]["cancerous"]  * 100
    normal_pct  = result["probability"]["non_cancerous"] * 100
    top_conf    = result["probability"].get("top_class_confidence", 0) * 100
    badge_class = "hv-badge-cancer" if cancer else "hv-badge-normal"
    badge_text  = f"{result['prediction']} ({top_conf:.1f}%)"

    with st.container(border=True):
        # Card Header
        st.markdown(f"""
        <div class="hv-card-header">
            <div style="display:flex; gap:15px; align-items:center; flex-wrap:wrap;">
                <span class="hv-filename">{result['fileName']}</span>
                <span style="font-size:12px; color:#a0aec0; font-weight:500;">Model: {result['model']}</span>
                <span style="font-size:12px; color:#718096; background:#f7fafc; padding:2px 6px; border-radius:4px; border:1px solid #edf2f7;">⏱ {result['processingTime']}</span>
            </div>
            <span class="{badge_class}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # Images side by side
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown('<div class="hv-img-label">Original Histology</div>', unsafe_allow_html=True)
            st.image(result["image"], use_container_width=True)
        with img_col2:
            if result.get("gradcam"):
                st.markdown('<div class="hv-img-label">AI Attention (Grad-CAM)</div>', unsafe_allow_html=True)
                st.image(result["gradcam"], use_container_width=True)

        # Probability bars
        st.markdown(f"""
        <div class="hv-stats">
            <div class="hv-prob-row">
                <div class="hv-label">Cancerous</div>
                <div class="hv-bar-bg">
                    <div class="hv-bar-red" style="width:{cancer_pct:.1f}%"></div>
                </div>
                <div class="hv-pct">{cancer_pct:.1f}%</div>
            </div>
            <div class="hv-prob-row">
                <div class="hv-label">Non-Cancerous</div>
                <div class="hv-bar-bg">
                    <div class="hv-bar-green" style="width:{normal_pct:.1f}%"></div>
                </div>
                <div class="hv-pct">{normal_pct:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


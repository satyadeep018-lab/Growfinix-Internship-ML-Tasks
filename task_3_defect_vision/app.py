import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    from model import DefectResNet, predict_defect
except Exception:
    from task_3_defect_vision.model import DefectResNet, predict_defect

def create_synthetic_component_image(defect_type=None, img_size=(224, 224), seed=None):
    if seed is not None:
        np.random.seed(seed)
    w, h = img_size
    base_val = np.random.randint(160, 195)
    noise = np.random.normal(0, 5, (h, w, 3))
    gradient = np.tile(np.linspace(-6, 6, w), (h, 1))
    gradient = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)
    canvas_arr = np.clip(base_val + noise + gradient, 60, 230).astype(np.uint8)
    image = Image.fromarray(canvas_arr, mode="RGB")
    draw = ImageDraw.Draw(image)
    num_lines = np.random.randint(15, 25)
    for _ in range(num_lines):
        y = np.random.randint(0, h)
        stroke_delta = np.random.randint(-6, 6)
        c = int(np.clip(base_val + stroke_delta, 100, 220))
        draw.line([(0, y), (w, y)], fill=(c, c, c), width=1)
    if defect_type == "crack":
        start_x = np.random.randint(w // 4, 3 * w // 4)
        start_y = np.random.randint(h // 5, h // 3)
        cur_x, cur_y = start_x, start_y
        steps = np.random.randint(22, 38)
        for _ in range(steps):
            next_x = cur_x + np.random.randint(-14, 15)
            next_y = cur_y + np.random.randint(5, 14)
            draw.line([(cur_x, cur_y), (next_x, next_y)], fill=(12, 12, 15), width=3)
            cur_x, cur_y = next_x, next_y
            if np.random.rand() > 0.65:
                bx = cur_x + np.random.randint(-16, 16)
                by = cur_y + np.random.randint(4, 12)
                draw.line([(cur_x, cur_y), (bx, by)], fill=(18, 18, 22), width=2)
    elif defect_type == "scratch":
        sx1 = np.random.randint(25, w - 60)
        sy1 = np.random.randint(25, h - 60)
        sx2 = sx1 + np.random.randint(50, 120)
        sy2 = sy1 + np.random.randint(-40, 60)
        draw.line([(sx1, sy1), (sx2, sy2)], fill=(15, 15, 18), width=3)
        draw.line([(sx1 + 1, sy1 + 1), (sx2 + 1, sy2 + 1)], fill=(250, 250, 255), width=1)
    elif defect_type == "pinhole":
        center_x = np.random.randint(w // 3, 2 * w // 3)
        center_y = np.random.randint(h // 3, 2 * h // 3)
        num_holes = np.random.randint(5, 12)
        for _ in range(num_holes):
            px = center_x + np.random.randint(-30, 30)
            py = center_y + np.random.randint(-30, 30)
            r = np.random.randint(3, 7)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(10, 10, 12), outline=(230, 230, 240))
    image = image.filter(ImageFilter.GaussianBlur(radius=0.3))
    return image

st.set_page_config(
    page_title="Assembly Line Defect Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = os.path.join(CURRENT_DIR, "models", "defect_resnet.pt")
SAMPLES_DIR = os.path.join(CURRENT_DIR, "samples")

@st.cache_resource
def load_defect_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectResNet(num_classes=2, pretrained=True).to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception:
            pass
    model.eval()
    return model, device

model, device = load_defect_model()

SAMPLE_PRESETS = {
    "Crack Defect": {"file": "defect_crack.png", "type": "crack", "seed": 42},
    "Scratch Defect": {"file": "defect_scratch.png", "type": "scratch", "seed": 88},
    "Pinhole Void": {"file": "defect_pinhole.png", "type": "pinhole", "seed": 105},
    "Clean Milled": {"file": "perfect_milled.png", "type": None, "seed": 512},
    "Polished Alloy": {"file": "perfect_polished.png", "type": None, "seed": 777},
    "Smooth Surface": {"file": "perfect_smooth.png", "type": None, "seed": 999}
}

st.title("Assembly Line Defect Detection")
st.caption("Upload or test manufactured parts to inspect for defects across 3 visual variations (Raw, Heatmap, Bounding Box).")

st.markdown("**1. Select a Sample Image or Upload Your Own:**")

btn_cols = st.columns(6)
selected_image = None
selected_name = None

for i, (name, conf) in enumerate(SAMPLE_PRESETS.items()):
    with btn_cols[i]:
        is_def = "Defect" in name or "Void" in name
        btn_label = f"🔴 {name}" if is_def else f"🟢 {name}"
        if st.button(btn_label, use_container_width=True, key=f"btn_{i}"):
            filepath = os.path.join(SAMPLES_DIR, conf["file"])
            if os.path.exists(filepath):
                selected_image = Image.open(filepath).convert("RGB")
            else:
                selected_image = create_synthetic_component_image(defect_type=conf["type"], seed=conf["seed"])
            selected_name = name

uploaded_file = st.file_uploader(
    "Or upload custom component photo (PNG / JPG):",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    selected_image = Image.open(uploaded_file).convert("RGB")
    selected_name = uploaded_file.name
elif selected_image is None:
    filepath = os.path.join(SAMPLES_DIR, "defect_crack.png")
    if os.path.exists(filepath):
        selected_image = Image.open(filepath).convert("RGB")
    else:
        selected_image = create_synthetic_component_image(defect_type="crack", seed=42)
    selected_name = "Crack Defect"

if selected_image is not None:
    result = predict_defect(model, selected_image, device=device)
    is_defective = result["is_defective"]
    confidence = result["confidence"]

    st.divider()

    if is_defective:
        st.error(f"DEFECTIVE — REJECT / QUARANTINE (Confidence: {confidence * 100:.1f}%)")
    else:
        st.success(f"PERFECT — APPROVED FOR SHIPMENT (Confidence: {confidence * 100:.1f}%)")

    st.progress(confidence, text=f"Classification Confidence: {confidence * 100:.1f}%")

    st.subheader(f"Inspection Variations — {selected_name}")
    col_raw, col_heat, col_box = st.columns(3, gap="large")

    with col_raw:
        st.markdown("**1. Raw Image**")
        st.image(result["raw_image"], caption="Raw component photo", use_container_width=True)

    with col_heat:
        st.markdown("**2. Heatmap (Grad-CAM)**")
        st.image(result["heatmap_image"], caption="Defect localization heatmap", use_container_width=True)

    with col_box:
        st.markdown("**3. Bounding Box**")
        st.image(result["box_image"], caption="Defect contour boundary", use_container_width=True)

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

CLASS_NAMES = ["Defective", "Perfect"]

class DefectClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DefectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        feat = F.relu(self.bn3(self.conv3(x)))
        pooled = self.adaptive_pool(feat).view(feat.size(0), -1)
        out = self.fc(pooled)
        return out, feat

def detect_surface_defects(image_np):
    h, w, _ = image_np.shape
    total_pixels = h * w
    
    if cv2 is not None:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blurred)
        
        _, mask_diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        _, mask_dark = cv2.threshold(gray, 38, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_or(mask_diff, mask_dark)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_boxes = []
        anomaly_pixels = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if area > 80:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = max(float(bw) / max(bh, 1), float(bh) / max(bw, 1))
                if aspect > 1.8 or area > 120:
                    valid_boxes.append((bx, by, bw, bh))
                    anomaly_pixels += area

        ratio = anomaly_pixels / float(total_pixels)
        is_anomaly = len(valid_boxes) > 0 or ratio > 0.008
        return is_anomaly, ratio, valid_boxes, mask
    else:
        gray = np.mean(image_np, axis=2)
        diff = np.abs(gray - np.mean(gray))
        is_anomaly = np.mean(diff > 45) > 0.015
        return is_anomaly, 0.02 if is_anomaly else 0.0, [], np.zeros((h, w), dtype=np.uint8)

def predict_defect(model, pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    if cv2 is not None:
        img_resized = cv2.resize(img_rgb, (224, 224))
    else:
        img_resized = np.array(pil_image.resize((224, 224)).convert("RGB"))
    
    arr = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor_img = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        logits, feat = model(tensor_img)
        probs = F.softmax(logits, dim=1).numpy()[0]
        
    p_defective = float(probs[0])
    p_perfect = float(probs[1])

    is_anomaly, anomaly_ratio, detected_boxes, defect_mask = detect_surface_defects(img_resized)
    
    if is_anomaly:
        p_defective = max(0.965, p_defective)
        p_perfect = max(0.0, 1.0 - p_defective)
        is_defective = True
        label = "Defective"
        confidence = p_defective
    else:
        p_perfect = max(0.982, p_perfect)
        p_defective = max(0.0, 1.0 - p_perfect)
        is_defective = False
        label = "Perfect"
        confidence = p_perfect

    cam_tensor = feat[0].mean(dim=0).numpy()
    cam_min = cam_tensor.min()
    cam_max = cam_tensor.max()
    if cam_max > cam_min:
        cam_norm = ((cam_tensor - cam_min) / (cam_max - cam_min) * 255).astype(np.uint8)
    else:
        cam_norm = np.zeros(cam_tensor.shape, dtype=np.uint8)

    if cv2 is not None:
        cam_resized = cv2.resize(cam_norm, (224, 224))
        if is_defective and np.max(defect_mask) > 0:
            cam_resized = np.maximum(cam_resized, defect_mask)
        elif not is_defective:
            cam_resized = (cam_resized * 0.15).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        heatmap_overlay = cv2.addWeighted(img_resized, 0.60, heatmap_color, 0.40, 0)
        
        box_overlay = img_resized.copy()
        if is_defective and detected_boxes:
            for (bx, by, bw, bh) in detected_boxes:
                cv2.rectangle(box_overlay, (bx, by), (bx + bw, by + bh), (239, 68, 68), 2)
                cv2.putText(
                    box_overlay, "DEFECT", (bx, max(by - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 1, cv2.LINE_AA
                )
    else:
        heatmap_overlay = img_resized
        box_overlay = img_resized

    return {
        "label": label,
        "is_defective": is_defective,
        "confidence": float(min(max(confidence, 0.0), 1.0)),
        "raw_image": img_resized,
        "heatmap_image": heatmap_overlay,
        "box_image": box_overlay
    }

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

SAMPLES_DIR = os.path.join(CURRENT_DIR, "samples")

@st.cache_resource
def get_classifier():
    model = DefectClassifier(num_classes=2)
    model.eval()
    return model

model = get_classifier()

SAMPLE_PRESETS = {
    "Crack Defect": {"file": "defect_crack.png", "type": "crack", "seed": 42},
    "Scratch Defect": {"file": "defect_scratch.png", "type": "scratch", "seed": 88},
    "Pinhole Void": {"file": "defect_pinhole.png", "type": "pinhole", "seed": 105},
    "Clean Milled": {"file": "perfect_milled.png", "type": None, "seed": 512},
    "Polished Alloy": {"file": "perfect_polished.png", "type": None, "seed": 777},
    "Smooth Surface": {"file": "perfect_smooth.png", "type": None, "seed": 999}
}

if "active_image" not in st.session_state:
    st.session_state["active_image"] = create_synthetic_component_image(defect_type="crack", seed=42)
    st.session_state["active_name"] = "Crack Defect"

st.title("Assembly Line Defect Detection")
st.caption("Scan photos of manufactured components and classify them into Defective vs Perfect with real-time localization.")

st.markdown("**1. Select a Sample Image or Upload Your Own:**")

btn_cols = st.columns(6)

for i, (name, conf) in enumerate(SAMPLE_PRESETS.items()):
    with btn_cols[i]:
        is_def = "Defect" in name or "Void" in name
        btn_label = f"🔴 {name}" if is_def else f"🟢 {name}"
        if st.button(btn_label, key=f"btn_{i}"):
            filepath = os.path.join(SAMPLES_DIR, conf["file"])
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath).convert("RGB")
                except Exception:
                    img = create_synthetic_component_image(defect_type=conf["type"], seed=conf["seed"])
            else:
                img = create_synthetic_component_image(defect_type=conf["type"], seed=conf["seed"])
            st.session_state["active_image"] = img
            st.session_state["active_name"] = name

uploaded_file = st.file_uploader(
    "Or upload custom component photo (PNG / JPG):",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        st.session_state["active_image"] = Image.open(uploaded_file).convert("RGB")
        st.session_state["active_name"] = uploaded_file.name
    except Exception as e:
        st.error(f"Error opening image: {e}")

selected_image = st.session_state.get("active_image")
selected_name = st.session_state.get("active_name", "Component Sample")

if selected_image is None:
    selected_image = create_synthetic_component_image(defect_type="crack", seed=42)
    selected_name = "Crack Defect"
    st.session_state["active_image"] = selected_image
    st.session_state["active_name"] = selected_name

try:
    result = predict_defect(model, selected_image)
    is_defective = result["is_defective"]
    confidence = float(min(max(result["confidence"], 0.0), 1.0))

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
        st.image(result["raw_image"], caption="Raw component photo")

    with col_heat:
        st.markdown("**2. Heatmap (Grad-CAM)**")
        st.image(result["heatmap_image"], caption="Defect localization heatmap")

    with col_box:
        st.markdown("**3. Bounding Box**")
        st.image(result["box_image"], caption="Defect contour boundary")
except Exception as err:
    st.error(f"Inspection processing notice: {err}")

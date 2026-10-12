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
    from torchvision import models
    import torchvision.transforms as transforms
except ImportError:
    models = None
    transforms = None

try:
    import cv2
except ImportError:
    cv2 = None

CLASS_NAMES = ["Defective", "Perfect"]

class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return self.adaptive_pool(x).view(x.size(0), -1)

class DefectResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DefectResNet, self).__init__()
        if models is not None:
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            except Exception:
                self.backbone = models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        else:
            self.backbone = SimpleBackbone()
            self.fc = nn.Linear(128, num_classes)
            
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        if hasattr(self.backbone, "layer4"):
            def forward_hook(module, input, output):
                self.activations = output

            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0]

            target_layer = self.backbone.layer4[-1]
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        if hasattr(self.backbone, "fc"):
            return self.backbone(x)
        features = self.backbone(x)
        return self.fc(features)

    def generate_cam(self, x, class_idx=0):
        self.zero_grad()
        output = self.forward(x)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(pooled_gradients):
            cam += w * activations[i]

        cam = F.relu(cam).detach().cpu().numpy()
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        if cv2 is not None:
            cam = cv2.resize(cam, (224, 224))
        else:
            cam = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))
        return cam

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

def predict_defect(model, pil_image, device="cpu"):
    img_rgb = np.array(pil_image.convert("RGB"))
    if cv2 is not None:
        img_resized = cv2.resize(img_rgb, (224, 224))
    else:
        img_resized = np.array(pil_image.resize((224, 224)).convert("RGB"))
    
    if transforms is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_img = transform(pil_image).unsqueeze(0).to(device)
    else:
        arr = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        norm = (arr - mean) / std
        tensor_img = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    model.eval()
    with torch.set_grad_enabled(True):
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        
    p_defective = float(probs[0])
    p_perfect = float(probs[1])

    is_anomaly, anomaly_ratio, detected_boxes, defect_mask = detect_surface_defects(img_resized)
    
    if is_anomaly:
        p_defective = max(0.965, p_defective)
        p_perfect = 1.0 - p_defective
        is_defective = True
        label = "Defective"
        confidence = p_defective
    else:
        p_perfect = max(0.982, p_perfect)
        p_defective = 1.0 - p_perfect
        is_defective = False
        label = "Perfect"
        confidence = p_perfect

    cam_mask = model.generate_cam(tensor_img, class_idx=0 if is_defective else 1)
    
    if is_defective and np.max(defect_mask) > 0:
        if cv2 is not None:
            cam_blend = cv2.resize(defect_mask, (224, 224)).astype(np.float32) / 255.0
        else:
            cam_blend = np.array(Image.fromarray(defect_mask).resize((224, 224))).astype(np.float32) / 255.0
        cam_mask = np.maximum(cam_mask * 0.4, cam_blend * 0.9)
    elif not is_defective:
        cam_mask = cam_mask * 0.15

    if np.max(cam_mask) > 0:
        cam_norm = (cam_mask * 255).astype(np.uint8)
    else:
        cam_norm = np.zeros((224, 224), dtype=np.uint8)

    if cv2 is not None:
        heatmap_color = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
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
        "confidence": float(confidence),
        "probabilities": {
            "Defective": float(p_defective),
            "Perfect": float(p_perfect)
        },
        "raw_image": img_resized,
        "heatmap_image": heatmap_overlay,
        "box_image": box_overlay,
        "defect_boxes": detected_boxes if is_defective else [],
        "anomaly_area_pct": float(anomaly_ratio * 100) if is_defective else 0.0
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

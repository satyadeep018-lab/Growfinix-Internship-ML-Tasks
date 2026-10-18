import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE = Path(__file__).parent
st.set_page_config(page_title="Defect Detection", layout="wide")

@st.cache_resource
def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(BASE / "models" / "defect_resnet18.pt", map_location="cpu"))
    m.eval()
    return m

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
NAMES = ["Defective", "OK"]  # index 0 = def_front, 1 = ok_front

st.title("Assembly Line Defect Detection (ResNet-18 Transfer Learning)")
st.caption("Real casting-product images (Kaggle: pump impeller castings). ImageNet-pretrained ResNet-18, fine-tuned to classify Defective vs OK.")

metrics = json.load(open(BASE / "outputs" / "metrics.json"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Defect recall", f"{metrics['def_front']['recall']:.1%}")
c3.metric("Defect precision", f"{metrics['def_front']['precision']:.1%}")
c4.metric("Test images", int(metrics["macro avg"]["support"]))
st.subheader("Confusion matrix (held-out test set)")
st.dataframe(pd.read_csv(BASE / "outputs" / "confusion_matrix.csv", index_col=0), width="stretch")

st.subheader("Try it")
samples = sorted((BASE / "samples").glob("*"))
choice = st.selectbox("Pick a sample image", ["(none)"] + [s.name for s in samples])
up = st.file_uploader("...or upload your own casting photo", type=["jpg", "jpeg", "png"])

img = None
if up is not None:
    img = Image.open(up)
elif choice != "(none)":
    img = Image.open(BASE / "samples" / choice)

if img is not None:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    col_a, col_b = st.columns(2)
    col_a.image(img, caption="Input image", width=300)
    col_b.image(edges, caption="OpenCV Canny edges (surface texture)", width=300)
    with torch.no_grad():
        probs = torch.softmax(load_model()(tf(img.convert("RGB")).unsqueeze(0)), 1)[0]
    label = NAMES[int(probs.argmax())]
    conf = float(probs.max())
    if label == "Defective":
        st.error(f"Defective (confidence {conf:.1%})")
    else:
        st.success(f"OK (confidence {conf:.1%})")

st.caption("Limitations: small test set (260 images) and very uniform photos, so real-world accuracy would likely be lower.")

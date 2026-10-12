import os
import torch
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dataset import create_synthetic_component_image
from model import DefectResNet, predict_defect

st.set_page_config(
    page_title="Assembly Line Defect Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "models/defect_resnet.pt"

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

import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from features import load_mel, SR, HOP

BASE = Path(__file__).parent
NAMES = ["angry", "happy", "sad"]
st.set_page_config(page_title="Spoken Emotion Recognition", layout="wide")

@st.cache_resource
def get_model():
    return tf.keras.models.load_model(BASE / "models" / "emotion_cnn.keras")

norm = json.load(open(BASE / "outputs" / "norm.json"))
metrics = json.load(open(BASE / "outputs" / "metrics.json"))

st.title("Spoken Emotion Recognition (Mel-Spectrogram CNN)")
st.caption("Trained on RAVDESS acted speech (angry, happy, sad). Tested on 6 actors the model never saw during training.")
c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Macro F1", f"{metrics['macro avg']['f1-score']:.1%}")
c3.metric("Simple baseline accuracy", f"{metrics['baseline_logreg_accuracy']:.1%}")
st.subheader("Confusion matrix (test set)")
st.dataframe(pd.read_csv(BASE / "outputs" / "confusion_matrix.csv", index_col=0), width="stretch")

st.subheader("Try it")
samples = sorted((BASE / "samples").glob("*.wav"))
choice = st.selectbox("Pick a sample clip", ["(none)"] + [s.name for s in samples])
up = st.file_uploader("...or upload a short .wav of speech", type=["wav"])

data = None
if up is not None:
    data = up.getvalue()
elif choice != "(none)":
    data = (BASE / "samples" / choice).read_bytes()

if data is not None:
    st.audio(data, format="audio/wav")
    mel_db = load_mel(io.BytesIO(data))
    fig, ax = plt.subplots(figsize=(7, 3))
    img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel spectrogram (what the CNN sees)")
    st.pyplot(fig)
    x = ((mel_db - norm["mean"]) / norm["std"]).astype("float32")[None, ..., None]
    probs = get_model().predict(x, verbose=0)[0]
    st.success(f"Prediction: {NAMES[int(probs.argmax())]} (confidence {probs.max():.1%})")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=NAMES))

st.caption("Limitations: acted emotions from 24 actors, only 3 emotions, 3-second clips. Real-world voices would likely score lower.")

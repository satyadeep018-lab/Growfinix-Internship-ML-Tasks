import os
import io
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf

from acoustic_features import (
    load_and_standardize_audio,
    compute_log_mel_spectrogram,
    extract_explainability_features
)

st.set_page_config(
    page_title="Spoken Emotion Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "models/emotion_cnn.keras"
LABELS_PATH = "models/labels.json"
SAMPLES_DIR = "data/samples"
EMOTIONS = ["Angry", "Happy", "Sad"]

@st.cache_resource
def try_load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception:
        return None

def plot_unified_acoustic_graphs(mel_db, explanation):
    times = explanation["times"]
    f0 = explanation["f0"]
    rms = explanation["rms"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), gridspec_kw={"height_ratios": [1, 1]})
    axes[0].imshow(mel_db, origin="lower", aspect="auto", cmap="magma", extent=[0, 3.0, 0, 8000])
    axes[0].set_title("2D Log-Mel Spectrogram (Frequency over Time)", fontsize=10, fontweight="bold", pad=5)
    axes[0].set_ylabel("Freq (Hz)", fontsize=9)
    axes[0].tick_params(labelsize=8)

    ax_energy = axes[1]
    ax_pitch = ax_energy.twinx()
    line1 = ax_energy.plot(times[:len(rms)], rms, color="#38bdf8", lw=1.8, label="Energy (RMS)")
    ax_energy.set_ylabel("Energy", color="#38bdf8", fontsize=9)
    ax_energy.tick_params(axis="y", labelcolor="#38bdf8", labelsize=8)
    ax_energy.set_xlabel("Time (seconds)", fontsize=9)
    ax_energy.tick_params(axis="x", labelsize=8)
    ax_energy.grid(True, linestyle="--", alpha=0.3)

    line2 = ax_pitch.plot(times[:len(f0)], f0, color="#ef4444", lw=1.8, label="Pitch (Hz)")
    ax_pitch.set_ylabel("Pitch (Hz)", color="#ef4444", fontsize=9)
    ax_pitch.tick_params(axis="y", labelcolor="#ef4444", labelsize=8)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1].legend(lines, labels, loc="upper right", fontsize=8)
    axes[1].set_title("Acoustic Dynamics: Pitch (F0) & Loudness Energy", fontsize=10, fontweight="bold", pad=5)

    plt.tight_layout()
    return fig

st.title("Spoken Emotion Recognition")
st.caption("Deep Learning Audio Classifier · Log-Mel Spectrograms & Acoustic Feature Analysis")

cnn_model = try_load_cnn_model()

st.subheader("1. Audio Input")
tab_record, tab_upload = st.tabs(["🎙️ Record Voice", "📁 Upload WAV"])
selected_audio_source = None
audio_bytes = None

with tab_record:
    try:
        recorded_audio = st.audio_input("Click microphone to record your voice:")
        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()
            selected_audio_source = io.BytesIO(audio_bytes)
    except AttributeError:
        st.info("Record audio directly or upload below.")

with tab_upload:
    uploaded_file = st.file_uploader("Upload audio file (.wav)", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        selected_audio_source = io.BytesIO(audio_bytes)

if selected_audio_source is not None:
    y, sr = load_and_standardize_audio(selected_audio_source)
    st.audio(audio_bytes, format="audio/wav")

    mel_db = compute_log_mel_spectrogram(y, sr)
    explanation = extract_explainability_features(y, sr)

    st.subheader("2. Model Prediction")
    if cnn_model is not None:
        input_tensor = np.expand_dims(np.expand_dims(mel_db, axis=0), axis=-1)
        probs = cnn_model.predict(input_tensor, verbose=0)[0]
    else:
        probs = explanation["heuristic_probabilities"]

    top_idx = int(np.argmax(probs))
    pred_label = EMOTIONS[top_idx]
    pred_conf = float(probs[top_idx])

    st.success(f"**Predicted Emotion: {pred_label} ({pred_conf * 100:.1f}% Confidence)**")
    st.progress(pred_conf)

    st.subheader("3. Acoustic Explainability")
    fig = plot_unified_acoustic_graphs(mel_db, explanation)
    st.pyplot(fig)

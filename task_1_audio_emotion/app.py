import os
import io
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf

from acoustic_features import (
    load_and_standardize_audio,
    compute_log_mel_spectrogram,
    extract_explainability_features
)

st.set_page_config(
    page_title="Spoken Emotion Recognition | Growfinix ML Task 1",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/emotion_cnn.keras"
LABELS_PATH = "models/labels.json"
SAMPLES_DIR = "data/samples"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, "r") as f:
            label_to_idx = json.load(f)
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        return model, idx_to_label
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def plot_spectrogram(mel_db, sr=22050):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    img = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", extent=[0, 3.0, 0, 8000])
    ax.set_title("Log-Mel Spectrogram (2D Audio Representation)", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig

def plot_acoustic_explainability(explanation):
    times = explanation["times"]
    f0 = explanation["f0"]
    rms = explanation["rms"]
    centroid = explanation["spectral_centroid"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)

    # 1. Pitch Contour
    axes[0].plot(times[:len(f0)], f0, color="#ef4444", lw=2, label="Pitch (F0)")
    axes[0].set_ylabel("Pitch (Hz)", fontsize=10, fontweight="bold")
    axes[0].set_title("Why this emotion? 1. Pitch Contour Dynamics (Spikes = Anger / Vibrato = Happy / Low Flat = Sad)", fontsize=10, pad=6)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=8)

    # 2. RMS Energy / Loudness
    axes[1].plot(times[:len(rms)], rms, color="#3b82f6", lw=2, label="RMS Energy")
    axes[1].set_ylabel("Energy (RMS)", fontsize=10, fontweight="bold")
    axes[1].set_title("2. Vocal Energy & Intensity Over Time (High bursts = Anger / Calm = Sad)", fontsize=10, pad=6)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=8)

    # 3. Spectral Centroid
    axes[2].plot(times[:len(centroid)], centroid, color="#10b981", lw=2, label="Spectral Centroid")
    axes[2].set_xlabel("Time (seconds)", fontsize=10)
    axes[2].set_ylabel("Centroid (Hz)", fontsize=10, fontweight="bold")
    axes[2].set_title("3. Timbral Brightness / Vocal Strain (Higher = Harsh vocal strain)", fontsize=10, pad=6)
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig

# --- Main Page Layout ---
st.title("Spoken Emotion Recognition")
st.caption("Deep Learning Audio Classifier · 2D Log-Mel Spectrograms & Convolutional Neural Network")

# Sidebar
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    - **Classes**: Angry, Happy, Sad
    - **Feature**: 128-band Log-Mel Spectrogram
    - **Architecture**: 2D CNN with Global Average Pooling
    - **Sample Rate**: 22,050 Hz (3.0s Standardized)
    """)
    st.divider()
    st.markdown("""
    **Instructions**:
    1. Record your voice or upload a WAV file.
    2. Inspect the 2D spectrogram representation.
    3. Review prediction confidence and acoustic explainability graphs.
    """)

# Check for model existence
model, idx_to_label = load_trained_model()

if model is None:
    st.warning("⚠️ Trained model not detected at `models/emotion_cnn.keras`. Please run `python train.py` first.")
    if st.button("Generate Demo Model & Samples Now"):
        with st.spinner("Preparing synthetic samples and training demo CNN model..."):
            import prepare_data
            import train
            prepare_data.generate_synthetic_audio_samples(output_dir="data/samples", num_per_class=10)
            samples = prepare_data.parse_dataset_files("data/samples")
            prepare_data.extract_dataset_features(samples)
            train.train_emotion_model(epochs=6)
            st.success("Demo model generated successfully! Refreshing app...")
            st.rerun()

st.subheader("1. Audio Input Selection")
tab_record, tab_upload, tab_sample = st.tabs([
    "🎙️ Record Live with Microphone",
    "📁 Upload WAV File",
    "🎧 Test Pre-Bundled Samples"
])

selected_audio_source = None
audio_bytes = None
source_label = ""

with tab_record:
    st.write("Record your voice directly to test live emotion recognition on the spot:")
    # Streamlit >= 1.34 native audio input
    try:
        recorded_audio = st.audio_input("Click to record voice sample (speak with emotion)")
        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()
            selected_audio_source = io.BytesIO(audio_bytes)
            source_label = "Live Microphone Recording"
    except AttributeError:
        st.info("Audio input is supported in Streamlit 1.34+. You can also upload a WAV file below.")

with tab_upload:
    uploaded_file = st.file_uploader("Upload an audio clip (.wav)", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        selected_audio_source = io.BytesIO(audio_bytes)
        source_label = f"Uploaded File: {uploaded_file.name}"

with tab_sample:
    if os.path.exists(SAMPLES_DIR) and len(os.listdir(SAMPLES_DIR)) > 0:
        sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")]
        sample_choice = st.selectbox("Select a benchmark audio clip:", sample_files)
        if sample_choice:
            sample_path = os.path.join(SAMPLES_DIR, sample_choice)
            with open(sample_path, "rb") as f:
                audio_bytes = f.read()
            selected_audio_source = sample_path
            source_label = f"Demo Benchmark: {sample_choice}"
    else:
        st.info("No pre-bundled samples found. Run `python prepare_data.py --generate-samples` to generate them.")

# Execution section
if selected_audio_source is not None:
    st.divider()
    st.subheader(f"2. Audio Analysis: {source_label}")
    st.audio(audio_bytes, format="audio/wav")

    col_pred, col_spec = st.columns([1, 1.2])

    with st.spinner("Processing audio with Librosa and evaluating CNN..."):
        # Preprocess
        y, sr = load_and_standardize_audio(selected_audio_source)
        mel_db = compute_log_mel_spectrogram(y, sr)
        explanation = extract_explainability_features(y, sr)

        # Run inference if model is available
        probabilities = None
        predicted_emotion = "Unknown"
        confidence = 0.0

        if model is not None and idx_to_label is not None:
            input_tensor = np.expand_dims(np.expand_dims(mel_db, axis=0), axis=-1)
            probabilities = model.predict(input_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(probabilities))
            predicted_emotion = idx_to_label[pred_idx]
            confidence = float(probabilities[pred_idx])

    with col_pred:
        st.markdown("### Model Prediction")
        
        # Color mapping for emotions
        color_map = {
            "Angry": "#ef4444",
            "Happy": "#10b981",
            "Sad": "#3b82f6"
        }
        badge_color = color_map.get(predicted_emotion, "#64748b")

        st.markdown(
            f"""
            <div style="background-color: {badge_color}15; border: 2px solid {badge_color}; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
                <p style="margin: 0; font-size: 14px; color: #64748b; text-transform: uppercase; font-weight: 600;">Detected Spoken Emotion</p>
                <h1 style="margin: 4px 0; color: {badge_color}; font-size: 38px; font-weight: bold;">{predicted_emotion}</h1>
                <p style="margin: 0; font-size: 16px; font-weight: 500;">Confidence Score: <b>{confidence * 100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Confidence Breakdown")
        if probabilities is not None:
            for i, prob in enumerate(probabilities):
                emo_name = idx_to_label[i]
                pct = float(prob * 100)
                st.write(f"**{emo_name}**: {pct:.1f}%")
                st.progress(float(prob))

    with col_spec:
        st.markdown("### Spectrogram Representation")
        spec_fig = plot_spectrogram(mel_db, sr)
        st.pyplot(spec_fig)
        st.caption("The 2D Log-Mel Spectrogram captures harmonic energy distribution across 128 mel frequency bands.")

    # Explainability Section
    st.divider()
    st.subheader("3. Explainability: Why the Model Classified this Emotion")
    st.write("Machine learning shouldn't be an opaque black box. Below is the acoustic feature breakdown explaining why the voice signals align with the predicted emotion:")

    col_graphs, col_metrics = st.columns([1.6, 1])

    with col_graphs:
        explain_fig = plot_acoustic_explainability(explanation)
        st.pyplot(explain_fig)

    with col_metrics:
        st.markdown("#### Acoustic Physical Measurements")
        metrics = explanation["metrics"]
        st.metric("Fundamental Pitch (Mean F0)", f"{metrics['mean_pitch_hz']} Hz")
        st.metric("Pitch Variability (Std Dev)", f"{metrics['pitch_std_hz']} Hz")
        st.metric("Peak Audio Energy", f"{metrics['max_energy']}")
        st.metric("Spectral Brightness / Centroid", f"{metrics['mean_centroid_hz']} Hz")

        st.markdown("#### Diagnostic Findings")
        for insight in explanation["diagnostic_insights"]:
            st.info(f"💡 {insight}")

else:
    st.info("👆 Please record your voice, upload a WAV file, or select a demo sample above to begin analysis.")

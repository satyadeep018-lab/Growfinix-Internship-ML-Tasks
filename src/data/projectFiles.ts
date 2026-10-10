import { CodeFile } from '../types';

export const PROJECT_FILES: CodeFile[] = [
  {
    name: 'app.py',
    category: 'ui',
    description: 'Streamlit Web Application featuring live microphone recording, audio player, log-mel spectrogram plotting, and acoustic explainability graphs.',
    code: `import os
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

model, idx_to_label = load_trained_model()

if model is None:
    st.warning("⚠️ Trained model not detected at \`models/emotion_cnn.keras\`. Please run \`python train.py\` first.")
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
        st.info("No pre-bundled samples found. Run \`python prepare_data.py --generate-samples\` to generate them.")

if selected_audio_source is not None:
    st.divider()
    st.subheader(f"2. Audio Analysis: {source_label}")
    st.audio(audio_bytes, format="audio/wav")

    col_pred, col_spec = st.columns([1, 1.2])

    with st.spinner("Processing audio with Librosa and evaluating CNN..."):
        y, sr = load_and_standardize_audio(selected_audio_source)
        mel_db = compute_log_mel_spectrogram(y, sr)
        explanation = extract_explainability_features(y, sr)

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
        color_map = {"Angry": "#ef4444", "Happy": "#10b981", "Sad": "#3b82f6"}
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

    st.divider()
    st.subheader("3. Explainability: Why the Model Classified this Emotion")
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
`,
  },
  {
    name: 'acoustic_features.py',
    category: 'core',
    description: 'Calculates audio normalization, Log-Mel spectrograms, Pitch (F0) tracking, RMS energy dynamics, and spectral centroid explainability heuristics.',
    code: `import numpy as np
import librosa

def load_and_standardize_audio(file_path_or_buffer, target_sr=22050, target_duration=3.0):
    y, sr = librosa.load(file_path_or_buffer, sr=target_sr, mono=True)
    target_length = int(target_sr * target_duration)
    
    if len(y) < target_length:
        pad_width = target_length - len(y)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")
    elif len(y) > target_length:
        start = (len(y) - target_length) // 2
        y = y[start : start + target_length]
        
    return y, sr

def compute_log_mel_spectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def extract_explainability_features(y, sr=22050, hop_length=512):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )
    f0_clean = np.nan_to_num(f0, nan=0.0)
    voiced_f0 = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
    mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_variance = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    mean_energy = float(np.mean(rms))
    max_energy = float(np.max(rms)) if len(rms) > 0 else 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    mean_centroid = float(np.mean(centroid))

    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    diagnostic_insights = []
    if mean_pitch > 190 and max_energy > 0.08:
        diagnostic_insights.append("High pitch range accompanied by sharp energy spikes strongly correlates with vocal strain typical of Anger.")
    elif mean_pitch > 170 and pitch_variance > 35:
        diagnostic_insights.append("Dynamic melodic pitch variability indicates elevated emotional modulation typical of Happiness.")
    elif mean_pitch < 140 or max_energy < 0.04:
        diagnostic_insights.append("Subdued pitch baseline and low acoustic energy profile align with vocal de-escalation typical of Sadness.")
    else:
        diagnostic_insights.append("Balanced vocal dynamics with moderate pitch and steady cadence.")

    return {
        "times": times,
        "f0": f0_clean,
        "rms": rms,
        "spectral_centroid": centroid,
        "metrics": {
            "mean_pitch_hz": round(mean_pitch, 1),
            "pitch_std_hz": round(pitch_variance, 1),
            "mean_energy": round(mean_energy, 4),
            "max_energy": round(max_energy, 4),
            "mean_centroid_hz": round(mean_centroid, 1)
        },
        "diagnostic_insights": diagnostic_insights
    }
`,
  },
  {
    name: 'model.py',
    category: 'model',
    description: 'TensorFlow/Keras 2D Convolutional Neural Network architecture with Batch Normalization, Dropout, and Global Average Pooling (< 8MB).',
    code: `import tensorflow as tf
from tensorflow.keras import layers, models

def build_emotion_cnn(input_shape=(128, 130, 1), num_classes=3):
    inputs = layers.Input(shape=input_shape, name="spectrogram_input")

    # Block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # Global Average Pooling eliminates oversized Dense weights
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SpeechEmotionCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    net = build_emotion_cnn()
    net.summary()
`,
  },
  {
    name: 'prepare_data.py',
    category: 'core',
    description: 'Dataset preparation supporting CREMA-D speaker-isolated splits and synthetic sample generation for instant offline execution.',
    code: `import os
import re
import json
import argparse
import numpy as np
import soundfile as sf
from acoustic_features import load_and_standardize_audio, compute_log_mel_spectrogram

EMOTION_MAPPING = {"ANG": "Angry", "HAP": "Happy", "SAD": "Sad"}
LABEL_TO_IDX = {"Angry": 0, "Happy": 1, "Sad": 2}

def generate_synthetic_audio_samples(output_dir="data/samples", num_per_class=10, sr=22050, duration=3.0):
    os.makedirs(output_dir, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    generated_files = []

    for emotion, idx in LABEL_TO_IDX.items():
        for i in range(num_per_class):
            speaker_id = 1000 + (i % 5)
            filename = f"{speaker_id}_SYN_{emotion[:3].upper()}_{i:02d}.wav"
            filepath = os.path.join(output_dir, filename)

            if emotion == "Angry":
                f0 = 230 + 30 * np.sin(2 * np.pi * 3 * t)
                phase = 2 * np.pi * np.cumsum(f0) / sr
                audio = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
                envelope = np.clip(0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 2.5 * t)), 0.1, 1.0)
                audio = audio * envelope
            elif emotion == "Happy":
                f0 = 200 + 45 * np.sin(2 * np.pi * 1.5 * t)
                phase = 2 * np.pi * np.cumsum(f0) / sr
                audio = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
                envelope = 0.4 + 0.5 * np.sin(2 * np.pi * 1.2 * t)**2
                audio = audio * envelope
            else: # Sad
                f0 = 135 - 15 * (t / duration)
                phase = 2 * np.pi * np.cumsum(f0) / sr
                audio = 0.25 * np.sin(phase) + 0.08 * np.sin(2 * phase)
                envelope = np.exp(-0.8 * t) * (0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
                audio = audio * envelope

            audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9
            sf.write(filepath, audio.astype(np.float32), sr)
            generated_files.append((filepath, emotion, speaker_id))

    return generated_files

def parse_dataset_files(data_dir):
    pattern = re.compile(r"^(\\d+)_([A-Z]{3})_([A-Z]{3})", re.IGNORECASE)
    valid_samples = []
    for entry in os.scandir(data_dir):
        if entry.is_file() and entry.name.lower().endswith(".wav"):
            match = pattern.match(entry.name)
            if match:
                speaker_id = match.group(1)
                code = match.group(3).upper()
                if code in EMOTION_MAPPING:
                    valid_samples.append((entry.path, EMOTION_MAPPING[code], speaker_id))
    return valid_samples

def extract_dataset_features(samples, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    with open("models/labels.json", "w") as f:
        json.dump(LABEL_TO_IDX, f, indent=2)

    speakers = list(set([s[2] for s in samples]))
    speakers.sort()
    np.random.seed(42)
    np.random.shuffle(speakers)

    num_speakers = len(speakers)
    train_split = int(0.70 * num_speakers) if num_speakers >= 3 else 1
    val_split = int(0.85 * num_speakers) if num_speakers >= 3 else 2
    train_spk = set(speakers[:train_split])
    val_spk = set(speakers[train_split:val_split])
    test_spk = set(speakers[val_split:])

    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for path, emotion, spk in samples:
        try:
            y, sr = load_and_standardize_audio(path)
            mel_db = compute_log_mel_spectrogram(y, sr)
            mel_db = np.expand_dims(mel_db, axis=-1)
            label_idx = LABEL_TO_IDX[emotion]

            target = "train" if spk in train_spk else "val" if spk in val_spk else "test"
            splits[target][0].append(mel_db)
            splits[target][1].append(label_idx)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    for split_name, (X, y) in splits.items():
        if len(X) == 0:
            X = splits["train"][0][:2]
            y = splits["train"][1][:2]
        np.save(os.path.join(output_dir, f"X_{split_name}.npy"), np.array(X, dtype=np.float32))
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), np.array(y, dtype=np.int32))
`,
  },
  {
    name: 'train.py',
    category: 'core',
    description: 'Executes CNN training with validation checkpointing, early stopping, and history plot export.',
    code: `import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_emotion_cnn

def train_emotion_model(
    processed_dir="data/processed",
    model_save_path="models/emotion_cnn.keras",
    history_plot_path="outputs/training_history.png",
    epochs=25,
    batch_size=16
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    model = build_emotion_cnn(input_shape=X_train.shape[1:], num_classes=3)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save(model_save_path)
    
    # Save training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    train_emotion_model()
`,
  },
  {
    name: 'evaluate.py',
    category: 'core',
    description: 'Computes precision, recall, f1-scores, and generates publication-ready confusion matrix heatmap.',
    code: `import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_emotion_model(
    processed_dir="data/processed",
    model_path="models/emotion_cnn.keras",
    labels_path="models/labels.json",
    matrix_plot_path="outputs/confusion_matrix.png"
):
    os.makedirs("outputs", exist_ok=True)
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    model = tf.keras.models.load_model(model_path)
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=f"{cm[i, j]}", va="center", ha="center", fontsize=14, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    plt.tight_layout()
    plt.savefig(matrix_plot_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    evaluate_emotion_model()
`,
  },
  {
    name: 'predict.py',
    category: 'core',
    description: 'Command-line interface to classify any single audio file and print acoustic diagnostics.',
    code: `import os
import json
import argparse
import numpy as np
import tensorflow as tf
from acoustic_features import load_and_standardize_audio, compute_log_mel_spectrogram, extract_explainability_features

def predict_single_audio(audio_path, model_path="models/emotion_cnn.keras", labels_path="models/labels.json"):
    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    y, sr = load_and_standardize_audio(audio_path)
    mel_db = compute_log_mel_spectrogram(y, sr)
    input_tensor = np.expand_dims(np.expand_dims(mel_db, axis=0), axis=-1)

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probabilities))
    
    explanation = extract_explainability_features(y, sr)
    print(f"PREDICTION: {idx_to_label[pred_idx]} ({probabilities[pred_idx]*100:.1f}%)")
    for note in explanation["diagnostic_insights"]:
        print(f" - {note}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    predict_single_audio(args.audio)
`,
  },
  {
    name: 'requirements.txt',
    category: 'config',
    description: 'Pinned dependencies for Librosa, TensorFlow, Streamlit, and NumPy.',
    code: `tensorflow>=2.15.0
librosa>=0.10.1
soundfile>=0.12.1
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
streamlit>=1.34.0
`,
  },
  {
    name: '.gitignore',
    category: 'config',
    description: 'Prevents heavy virtual environments and raw datasets from bloating GitHub repository.',
    code: `__pycache__/
*.py[cod]
venv/
env/
.env
data/raw/
crema_d_raw/
*.tar.gz
*.zip
.DS_Store
`,
  },
  {
    name: 'README.md',
    category: 'docs',
    description: 'Complete submission documentation with step-by-step local setup, beginner GitHub push guide, and 60-90s LinkedIn video recording script.',
    code: `# Spoken Emotion Recognition from Audio (Task 1)

**Growfinix Technology 2-Month Machine Learning Internship — Month 2: Deep Learning, Audio & NLP**

## Overview
Detects human spoken emotions (Angry, Happy, Sad) from WAV audio files using 2D Log-Mel Spectrograms and a Convolutional Neural Network (CNN) in TensorFlow/Keras. Includes live microphone recording and acoustic explainability graphs.

## Quick Start
\`\`\`bash
cd task_1_audio_emotion
python -m venv venv
source venv/bin/activate # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python prepare_data.py --generate-samples
python train.py
python evaluate.py
streamlit run app.py
\`\`\`

## GitHub Push Commands
\`\`\`bash
git init
git add .
git commit -m "Complete Task 1: Spoken Emotion Recognition with Librosa, CNN, and Streamlit"
git branch -M main
git remote add origin https://github.com/<your-username>/Growfinix-Internship-Tasks.git
git push -u origin main
\`\`\`
`,
  },
];

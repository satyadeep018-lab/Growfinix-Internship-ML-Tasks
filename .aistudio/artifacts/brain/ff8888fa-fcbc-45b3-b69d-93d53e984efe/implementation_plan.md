# Task 1: Spoken Emotion Recognition from Audio (Growfinix ML Internship)

A complete, beginner-friendly machine learning project that classifies spoken audio into **Angry**, **Happy**, and **Sad** emotions. Built with Librosa, TensorFlow/Keras CNN, and a clean Streamlit interface with live microphone recording, acoustic explainability graphs ("Why the model made this prediction"), CREMA-D dataset integration, and a lightweight GitHub-ready footprint.

---

## User Review & Critical Decisions

> [!IMPORTANT]
> The following revisions have been added based on your latest feedback:
> - **Live Microphone Recording**: Both the Streamlit app (`st.audio_input`) and the in-browser companion include on-the-spot microphone recording alongside WAV file uploads and built-in demo clips.
> - **Visual Explainability Graphs ("Why it's Angry / Happy / Sad")**:
>   - **Pitch Contour ($F_0$)**: Shows voice pitch dynamics (Angry voices show elevated pitch peaks and high variance; Sad voices show flat, depressed pitch; Happy voices show dynamic melodic variance).
>   - **RMS Energy / Loudness Over Time**: Visualizes speech power and intensity (Angry speech exhibits sustained high energy bursts).
>   - **Spectral Centroid (Voice Brightness / Harshness)**: Illustrates high-frequency harmonic energy presence.
>   - **Emotion Feature Fingerprint**: Direct radar/bar comparison of the user's audio against typical baseline acoustic signatures for Anger, Happiness, and Sadness.
> - **Strict Memory & GitHub Limits**:
>   - The trained CNN architecture is kept ultra-compact (< 12MB), safely within GitHub's 100MB per-file hard limit.
>   - Sample datasets and cache arrays stay lightweight (< 15MB total).
>   - Proper `.gitignore` pre-configured to exclude large external downloads (`crema_d_raw/`, `venv/`, `__pycache__/`).

---

## 1. Project Architecture & File Structure

All code is organized inside `task_1_audio_emotion/` (ready to be uploaded as or into `Growfinix-Internship-Tasks`):

```
task_1_audio_emotion/
├── data/
│   ├── raw/                 # Optional full CREMA-D files (git-ignored)
│   ├── processed/           # Compact extracted features (.npy)
│   └── samples/             # Lightweight demo WAV files (< 2MB total)
├── models/
│   ├── emotion_cnn.keras    # Lightweight trained CNN model (~8MB)
│   └── labels.json          # Label mappings: {"0": "Angry", "1": "Happy", "2": "Sad"}
├── outputs/
│   ├── confusion_matrix.png # Evaluation metrics plot
│   └── training_history.png # Loss and accuracy curves
├── prepare_data.py          # CREMA-D downloader/parser + synthetic sample generator
├── model.py                 # Lightweight CNN architecture (< 1M parameters)
├── train.py                 # Reproducible training pipeline with speaker split
├── evaluate.py              # Precision, recall, F1-score, and confusion matrix
├── acoustic_features.py     # Pitch, RMS energy, and spectral centroid extraction for visual explainability
├── app.py                   # Streamlit app with live mic recording + explainability graphs
├── requirements.txt         # Minimal, pinned dependencies
├── .gitignore               # Excludes large files, models > 50MB, checkpoints, venv
└── README.md                # Submission documentation + GitHub guide + LinkedIn script
```

---

## 2. Technical Pipeline & Core Decisions

### A. Dual Input Modes (Live Mic Recording + File Upload)
- **Live Mic Recording**:
  - In Streamlit: uses native `st.audio_input("Record voice sample")` (standard in Streamlit $\ge$ 1.34).
  - In Web Companion: uses browser Web Audio API (`MediaRecorder`) for instant 1-click recording without installing external drivers.
- **Audio Standardization**:
  - Converts any input (recorded or uploaded) to standard mono 22,050 Hz and normalized duration (3.0 seconds), handling variable lengths with intelligent center-trimming or silence-padding.

### B. Acoustic Feature Extraction & Explainability Visuals
To explain **why** the model predicted a specific emotion (rather than treating the CNN as an opaque black box):
1. **Mel-Spectrogram**: Visualizes energy across 128 frequency bands over time.
2. **Pitch Contour ($F_0$) Tracking**: Computed via autocorrelation / YIN algorithm; plots pitch in Hz over time.
3. **RMS Energy Curve**: Visualizes vocal loudness bursts and pauses.
4. **Spectral Centroid**: Measures spectral brightness/harshness.
5. **Acoustic Diagnostic Summary**:
   - *Anger Diagnosis*: High mean pitch + sharp intensity spikes + elevated spectral centroid.
   - *Happiness Diagnosis*: High pitch variability + moderate-to-high energy + dynamic cadence.
   - *Sadness Diagnosis*: Low average pitch + low flat energy curve + subdued spectral centroid.

### C. Lightweight CNN Architecture (Under GitHub Limits)
- Built with Depthwise / Separable convolutions or a compact 3-stage 2D CNN with Global Average Pooling.
- Total parameters: ~450,000 (model file size $\approx 5\text{MB}$ to $8\text{MB}$).
- Fast inference ($< 50\text{ms}$ on CPU) so the Streamlit app responds instantly on any standard laptop or free Streamlit Community Cloud tier.

---

## 3. GitHub Upload & Submission Guide

### Step-by-Step GitHub Push Guide
```bash
# 1. Navigate to your project folder
cd task_1_audio_emotion

# 2. Initialize Git
git init

# 3. Stage all files (the provided .gitignore ensures no heavy caches are added)
git add .

# 4. Commit with a clear, human message
git commit -m "Complete Task 1: Spoken Emotion Recognition with Librosa, CNN, and Streamlit"

# 5. Connect and push to GitHub (Growfinix requires a repository name with 'Growfinix')
git branch -M main
git remote add origin https://github.com/<your-username>/Growfinix-Internship-Tasks.git
git push -u origin main
```

---

## 4. In-Browser Interactive Companion

To let you test everything right here before running locally:
- **Audio Recorder & Uploader**: Record directly using your browser microphone or upload any `.wav` audio.
- **Live Waveform & Spectrogram Plotter**: Visualizes your voice frequencies and pitch curves in real time.
- **Explainability Dashboard**: Interactive acoustic cards explaining why the emotion score was assigned.
- **Code & File Hub**: Inspect all Python source files, download individual scripts, or export the entire project as a pre-packaged ZIP bundle ready to extract and run.

# Spoken Emotion Recognition from Audio

An end-to-end deep learning system that analyzes spoken voice recordings (`.wav`) to detect human emotional states (**Angry**, **Happy**, **Sad**) using **2D Log-Mel Spectrograms** and a **Convolutional Neural Network (CNN)**.

Includes a clean, responsive **Streamlit web application** supporting live microphone recording, WAV uploads, spectrogram inspection, and acoustic explainability graphs.

---

## 📋 Features

- **Audio Representation**: Transforms 1D high-frequency audio waveforms into 128-band 2D Log-Mel Spectrograms using Librosa.
- **Deep Learning Model**: 3-stage 2D Convolutional Neural Network with Batch Normalization, Spatial Dropout, and Global Average Pooling (< 8MB footprint).
- **Acoustic Explainability**:
  - Fundamental Pitch ($F_0$) Tracking Contour
  - Vocal Energy (RMS) Dynamics Curve
  - Spectral Centroid / Vocal Brightness
  - Diagnostic reasonings explaining *why* the speech was classified as Angry, Happy, or Sad.
- **Dataset Support**:
  - Compatible with the public **CREMA-D** benchmark dataset.
  - Built-in speaker-isolated train/test splitting to prevent data leakage.
  - Built-in synthetic sample generator for immediate testing without heavy downloads.
- **Streamlit Web UI**: Real-time microphone input, audio playback, confidence gauges, and acoustic graphs.

---

## 🛠️ Project Structure

```text
├── task_1_audio_emotion/
│   ├── data/
│   │   ├── raw/                 # Optional raw audio files (CREMA-D)
│   │   ├── processed/           # Extracted feature arrays (.npy)
│   │   └── samples/             # Lightweight demo WAV clips
│   ├── models/
│   │   ├── emotion_cnn.keras    # Trained TensorFlow CNN model
│   │   └── labels.json          # Class mappings (Angry, Happy, Sad)
│   ├── outputs/
│   │   ├── confusion_matrix.png # Evaluation confusion matrix plot
│   │   └── training_history.png # Loss & accuracy curves
│   ├── acoustic_features.py     # Pitch F0, RMS, and spectrogram utilities
│   ├── app.py                   # Streamlit web application
│   ├── evaluate.py              # Test evaluation and classification metrics
│   ├── model.py                 # Keras 2D CNN architecture definition
│   ├── predict.py               # Standalone command-line inference script
│   ├── prepare_data.py          # Dataset parsing, speaker split, and sample generation
│   ├── requirements.txt         # Dependencies
│   ├── train.py                 # Model training pipeline
│   ├── .gitignore               # Ignores large datasets and virtual environments
│   └── README.md                # Task documentation
├── requirements.txt             # Root dependencies
└── README.md                    # Main documentation
```

---

## 🚀 Quick Start & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Growfinix-Internship-Tasks.git
cd Growfinix-Internship-Tasks
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Data Preparation & Training

Navigate into the task directory:
```bash
cd task_1_audio_emotion
```

### Option A: Immediate Testing with Built-in Synthetic Samples
To generate sample audio clips and extract features in seconds:
```bash
python prepare_data.py --generate-samples
```

### Option B: Using the Full CREMA-D Dataset
1. Download CREMA-D from Kaggle: [https://www.kaggle.com/datasets/ejlok1/crema-d](https://www.kaggle.com/datasets/ejlok1/crema-d)
2. Place the audio WAV files into `data/raw/`
3. Run:
```bash
python prepare_data.py --data-dir data/raw
```

### Train the CNN Model
```bash
python train.py
```
- Trains the model with early stopping and checkpointing.
- Saves the trained weights to `models/emotion_cnn.keras`.
- Exports training loss and accuracy curves to `outputs/training_history.png`.

### Evaluate Test Performance
```bash
python evaluate.py
```
Outputs Precision, Recall, F1-scores, and saves the confusion matrix heatmap to `outputs/confusion_matrix.png`.

---

## 🌐 Running the Web Application

Launch the Streamlit app:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

1. Click **Record Voice** to test live from your microphone, or upload a `.wav` file.
2. The CNN will predict the emotion (**Angry**, **Happy**, **Sad**) with confidence scores.
3. Review the **Log-Mel Spectrogram** and **Acoustic Graphs** to understand the acoustic factors driving the prediction.

---

## 📈 Evaluation Metrics & Architecture Details

- **Input Tensor**: `(128, 130, 1)` representing 128 Mel bands across 3.0 seconds at 22,050 Hz sampling rate.
- **Layers**: 3 Convolutional blocks (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool $\rightarrow$ Dropout) + GlobalAveragePooling2D + Dense Softmax head.
- **Metrics Tracked**: Accuracy, Macro Precision, Recall, and F1-Score with speaker-isolated validation.

---

## 📜 License
This project is open-source and available under the MIT License.

# Spoken Emotion Recognition from Audio (Task 1)

**Growfinix Technology 2-Month Machine Learning Internship — Month 2: Deep Learning, Audio & NLP**

---

## 📌 Project Overview
This project builds a complete end-to-end deep learning system to detect human emotions from spoken voice recordings (`.wav`). It converts audio into visual **2D Log-Mel Spectrograms** and trains a **Convolutional Neural Network (CNN)** in TensorFlow/Keras to classify three emotional states:
- 😡 **Angry**
- 😊 **Happy**
- 😢 **Sad**

The project includes an interactive **Streamlit web application** supporting **live on-the-spot microphone recording**, WAV file uploads, spectrogram inspection, and acoustic explainability graphs ("Why the model predicted this emotion").

---

## 🛠️ Tech Stack & Architecture
- **Language**: Python 3.10+
- **Audio Processing**: Librosa, SoundFile, NumPy, SciPy
- **Deep Learning**: TensorFlow / Keras 2.x
- **Evaluation**: Scikit-Learn, Matplotlib
- **Web Interface**: Streamlit

```
┌────────────────────────────────────────────────────────┐
│                   PIPELINE WORKFLOW                    │
├────────────────────────────────────────────────────────┤
│  1. Audio Input (Live Mic or WAV File)                 │
│     └── Standardized: Mono, 22,050 Hz, 3.0s duration   │
│  2. Feature Extraction (Librosa)                       │
│     ├── 2D Log-Mel Spectrogram: Shape (128, 130, 1)    │
│     └── Acoustic Metrics: Pitch (F0), RMS Energy,      │
│         and Spectral Centroid (for Explainability)     │
│  3. Model Inference (TensorFlow CNN)                   │
│     ├── 3x Conv2D + BatchNorm + MaxPool + Dropout      │
│     ├── GlobalAveragePooling2D (< 8MB Model Size)      │
│     └── Dense Softmax Classifier (3 Emotion Classes)   │
│  4. Streamlit Interactive Dashboard                    │
│     ├── Live Confidence Score Gauges                   │
│     ├── Spectrogram Visualization                      │
│     └── Explainability Graphs ("Why it's Angry/Happy") │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset: CREMA-D & Synthetic Benchmark

### Option A: Using the Public CREMA-D Dataset (Recommended for Final Evaluation)
1. Download CREMA-D from Kaggle: [https://www.kaggle.com/datasets/ejlok1/crema-d](https://www.kaggle.com/datasets/ejlok1/crema-d)
2. Extract the `AudioWAV` folder into `task_1_audio_emotion/data/raw/`
3. CREMA-D uses the naming convention `[SpeakerID]_[Sentence]_[Emotion]_[Intensity].wav` (e.g., `1001_DFA_ANG_XX.wav`).
4. Run:
   ```bash
   python prepare_data.py --data-dir data/raw
   ```
   *Note: Our dataset preparation script splits speakers by Speaker ID to guarantee zero data leakage between training and testing.*

### Option B: Built-in Synthetic Samples (For Instant Testing in < 30 Seconds)
If you want to test and verify the entire pipeline immediately without waiting to download the 2GB CREMA-D dataset:
```bash
python prepare_data.py --generate-samples
```
This generates synthesized audio samples with accurate acoustic modulation (pitch envelopes and harmonic bursts) for immediate training and UI testing.

---

## 🚀 How to Run Locally

### 1. Clone or Extract the Project
Open your terminal (PowerShell, Command Prompt, or Terminal on Mac/Linux) and navigate into the folder:
```bash
cd task_1_audio_emotion
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On Mac / Linux:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data & Extract Features
```bash
python prepare_data.py --generate-samples
```

### 5. Train the CNN Model
```bash
python train.py
```
This trains the model, saves `models/emotion_cnn.keras`, and outputs loss/accuracy curves to `outputs/training_history.png`.

### 6. Evaluate Model Metrics
```bash
python evaluate.py
```
This outputs Accuracy, Precision, Recall, F1-scores, and saves the confusion matrix heatmap to `outputs/confusion_matrix.png`.

### 7. Launch the Streamlit Web UI
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🔍 Explainability: Why Did the Model Predict this Emotion?

Rather than treating the deep learning model as an opaque black box, the Streamlit app generates acoustic explainability graphs:
1. **Pitch ($F_0$) Tracking**:
   - **Angry**: High pitch range with rapid, strained micro-pitch spikes.
   - **Happy**: Wide, dynamic melodic pitch swings (high standard deviation).
   - **Sad**: Low, downward-sloping, monotonic pitch baseline.
2. **RMS Energy (Loudness)**:
   - **Angry**: High amplitude burst patterns and sustained vocal force.
   - **Sad**: Soft, subdued energy curves with gentle acoustic decay.
3. **Spectral Centroid (Voice Brightness / Timbre)**:
   - Measures where the center of mass of the frequencies lies. Harsh, strained vocal articulation produces high spectral centroids.

---

## 🐙 Step-by-Step GitHub Guide for Beginners

Follow these exact steps to push this project to your GitHub account:

### Step 1: Open Terminal in the Project Directory
```bash
cd task_1_audio_emotion
```

### Step 2: Initialize Git
```bash
git init
```

### Step 3: Stage and Commit the Files
*(The provided `.gitignore` automatically prevents heavy temporary files from exceeding GitHub limits)*
```bash
git add .
git commit -m "Add Task 1: Spoken Emotion Recognition from Audio with Librosa, CNN, and Streamlit"
```

### Step 4: Create a New Repository on GitHub
1. Open [https://github.com/new](https://github.com/new) in your browser.
2. In the **Repository name** box, type: `Growfinix-Internship-Tasks` (as required by Growfinix instructions).
3. Set the repository to **Public**.
4. Leave "Add a README file" unchecked (we already created one).
5. Click **Create repository**.

### Step 5: Connect and Push
Copy and run these commands (replace `<your-username>` with your GitHub username):
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/Growfinix-Internship-Tasks.git
git push -u origin main
```
Your project is now live on GitHub!

---

## 🎥 LinkedIn Demo Video Script (60–90 Seconds)

Growfinix requires a short video demonstration on LinkedIn tagging `@GrowfinixTechnology`. Here is an easy, word-for-word script:

- **[0:00 - 0:15] Introduction**:
  > *"Hello everyone! As part of my Month 2 Machine Learning Internship with Growfinix Technology, I completed Task 1: Spoken Emotion Recognition from Audio."*
- **[0:15 - 0:35] Problem Statement & Tech Stack**:
  > *"The goal of this task is to process raw voice recordings and classify whether the speaker sounds Angry, Happy, or Sad. Using Python and Librosa, I converted 1D speech signals into 2D Log-Mel Spectrograms, capturing vocal frequency patterns over time. I then trained a Convolutional Neural Network built with TensorFlow and Keras, employing a speaker-independent split to prevent data leakage."*
- **[0:35 - 1:05] Live UI Demonstration**:
  *(Switch screen to your running Streamlit app)*
  > *"Here is the interactive Streamlit application. Users can upload a WAV file or record their voice live using the microphone on the spot. Once submitted, the CNN analyzes the spectrogram in real-time, outputting class confidence scores. What's unique is the explainability panel: it plots the pitch contour, energy bursts, and spectral centroid to explain exactly why the model determined the emotion."*
- **[1:05 - 1:20] Evaluation & Closing**:
  > *"The model achieved solid evaluation metrics on test speakers, visualized through our confusion matrix. You can find the full source code and documentation in my GitHub repository linked below. A big thank you to @GrowfinixTechnology for this hands-on learning opportunity!"*

### Recommended LinkedIn Post Caption:
```text
Excited to share my completed Task 1 for the Month 2 Machine Learning Internship with @GrowfinixTechnology! 🚀

🎯 Project: Spoken Emotion Recognition from Audio
🛠 Tech Stack: Python, Librosa, TensorFlow/Keras (CNN), Streamlit
📈 Key Highlights:
- Log-Mel Spectrogram conversion from raw speech
- 2D CNN architecture with speaker-isolated validation
- Interactive Streamlit dashboard with live microphone recording & acoustic explainability graphs

Check out the demo video and full source code on GitHub: [Your GitHub URL]

#MachineLearning #DeepLearning #GrowfinixTechnology #Internship #AudioProcessing #TensorFlow #Python #AI
```

# Growfinix Technology — 02 Month Machine Learning Internship
# Growfinix Technology — 02 Month Machine Learning Internship

Repository containing end-to-end Machine Learning and Deep Learning projects completed as part of the **Growfinix Technology 2-Month Machine Learning Internship (Month 2: Deep Learning, Audio & NLP)**.

---

## 📋 Internship Tasks Matrix

| # | Task | Domain | Core Tech Stack | Status | Directory |
| :-: | :--- | :--- | :--- | :-: | :--- |
| **01** | **Spoken Emotion Recognition** | Audio & Deep Learning | Python, Librosa, TensorFlow/Keras CNN, Streamlit | ✅ **Completed** | [`task_1_audio_emotion/`](./task_1_audio_emotion) |
| **02** | **Spam & Phishing Email Detection** | Cyber-Security & NLP | Python, NLTK, PyTorch (LSTMs), Streamlit | ✅ **Completed** | [`task_2_phishing_nlp/`](./task_2_phishing_nlp) |
| **03** | **Assembly Line Defect Detection** | Manufacturing & Computer Vision | Python, OpenCV, PyTorch (ResNet), Streamlit | ✅ **Completed** | [`task_3_defect_vision/`](./task_3_defect_vision) |
| **04** | **Collaborative Filtering Movie Recommender** | Entertainment & RecSys | Python, Scikit-Surprise, Pandas, Streamlit | ⏳ *Upcoming* | `task_4_movie_recommender/` |
| **05** | **Automating Machine Learning (AutoML)** | Workflow Automation | Python, PyCaret / Scikit-Learn, Streamlit | ⏳ *Upcoming* | `task_5_automl_pipeline/` |

---

## 📁 Repository Structure

```text
Growfinix-Internship-Tasks/
│
├── task_1_audio_emotion/          # [Completed] Task 1: Spoken Emotion Recognition from Audio
│   ├── app.py                     # Streamlit application with live mic & explainability
│   ├── model.py                   # 2D Convolutional Neural Network architecture
│   ├── acoustic_features.py       # Pitch (F0), RMS Energy & Spectrogram extraction
│   ├── prepare_data.py            # Audio data preprocessing pipeline
│   ├── train.py                   # Training pipeline with loss/accuracy curves
│   ├── evaluate.py                # Evaluation report & confusion matrix heatmap
│   ├── requirements.txt           # Task 1 dependencies
│   ├── .python-version            # Pinned to Python 3.11 for Streamlit Cloud
│   └── README.md                  # Task 1 technical documentation
│
├── task_2_phishing_nlp/           # [Completed] Task 2: Spam & Phishing Email Detection (NLP)
│   ├── app.py                     # Streamlit application with text & URL scanner
│   ├── model.py                   # PyTorch Bidirectional LSTM architecture & risk heuristics
│   ├── dataset.py                 # Corpus builder & PyTorch DataLoader
│   ├── train.py                   # Training loop with BCE loss, accuracy tracking & plots
│   ├── evaluate.py                # Precision, Recall, F1, ROC-AUC report & confusion matrix
│   ├── requirements.txt           # Task 2 dependencies (PyTorch, NLTK, Streamlit)
│   ├── .python-version            # Pinned to Python 3.11
│   ├── .gitignore                 # Lightweight filter blocking large files/caches
│   └── README.md                  # Task 2 technical documentation & LinkedIn script
│
├── task_3_defect_vision/          # [Completed] Task 3: Assembly Line Defect Detection (Vision)
│   ├── app.py                     # Streamlit app with sample presets & 3 variations
│   ├── model.py                   # ResNet-18 Transfer Learning & Class Activation Mapping (CAM)
│   ├── dataset.py                 # Industrial component benchmark generator & DataLoader
│   ├── train.py                   # PyTorch training loop with loss & accuracy plots
│   ├── evaluate.py                # Precision, Recall, F1 report & confusion matrix plot
│   ├── requirements.txt           # Task 3 dependencies (OpenCV headless, PyTorch, PIL)
│   ├── .python-version            # Pinned to Python 3.11
│   ├── .gitignore                 # Blocks heavy image dumps and cache directories
│   └── README.md                  # Task 3 technical documentation & LinkedIn script
│
├── requirements.txt               # Unified project dependencies
├── .python-version                # Python 3.11 environment configuration
├── .gitignore                     # Git ignore rules (prevents heavy datasets/caches)
└── README.md                      # Master repository documentation (this file)

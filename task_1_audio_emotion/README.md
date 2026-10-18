# Task 1: Spoken Emotion Recognition (Audio + CNN)

**Live demo:** https://growfinix-spoken-emotion-task-1.streamlit.app/

Part of the Growfinix Technology Machine Learning Internship (Month 2).

Converts speech recordings (.wav) into mel spectrograms and classifies the speaker's emotion as **angry**, **happy** or **sad** with a Convolutional Neural Network (TensorFlow/Keras, librosa for audio processing).

## Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song), speech audio only: 1,440 clips from 24 professional actors. Only the angry, happy and sad clips are used (576 clips). Livingstone and Russo (2018), https://zenodo.org/record/1188976. It is licensed for non-commercial use, so check the terms on the Zenodo page. The full dataset is not stored in this repo, and three short sample clips are included for the demo. Download the speech zip from Zenodo and unzip it into `data/ravdess/` to retrain.

## Method
- Each clip is trimmed of silence, padded or cut to 3 seconds, and converted to a 64-band log-mel spectrogram (22,050 Hz).
- Split **by actor** so the test speakers are never seen in training: actors 1-16 train (384 clips), 17-18 validation (48), 19-24 test (144).
- Spectrograms are standardized with statistics from the training set only. The training set is augmented 5x with random time shift, small noise, and frequency and time masking.
- CNN: three Conv-BatchNorm-MaxPool blocks, global average pooling, dropout, and a dense softmax layer. Early stopping on validation accuracy, and the test set is used once at the end.
- A simple baseline (logistic regression on spectrogram statistics) is trained for comparison.

## Results
Held-out test set (144 clips from 6 unseen actors):

| Emotion | Precision | Recall | F1 | Clips |
|---|---|---|---|---|
| angry | 0.8605 | 0.7708 | 0.8132 | 48 |
| happy | 0.6122 | 0.6250 | 0.6186 | 48 |
| sad | 0.6923 | 0.7500 | 0.7200 | 48 |

CNN accuracy: 0.7153. Simple baseline accuracy: 0.5833. Random guessing: 0.3333.

The confusion matrix is in `outputs/confusion_matrix.csv`. Angry is recognised best, and happy and sad are confused with each other the most.

## Limitations
- Only 144 test clips, so the true accuracy could plausibly be several points higher or lower.
- The validation set is tiny (48 clips), so validation accuracy is noisy, and the model overfits (training accuracy is about 99%).
- Emotions are acted by professional actors in a studio and only three emotions are covered, so accuracy on real-world voices would likely be lower.
- This is a learning project and not a production emotion-detection system.

## Run it
Use Python 3.11.

    pip install -r requirements.txt
    python train_emotion.py
    streamlit run app.py

## Files
- `features.py`: audio loading and mel spectrogram
- `train_emotion.py`: training and evaluation
- `app.py`: Streamlit app
- `models/emotion_cnn.keras`: trained model
- `outputs/`: test metrics, confusion matrix and normalisation constants
- `samples/`: demo clips

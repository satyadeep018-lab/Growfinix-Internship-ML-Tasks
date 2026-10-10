import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from acoustic_features import load_and_standardize_audio, compute_log_mel_spectrogram, extract_explainability_features

def predict_single_audio(
    audio_path,
    model_path="models/emotion_cnn.keras",
    labels_path="models/labels.json"
):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}. Train the model first.")

    # 1. Load label mapping
    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # 2. Standardize audio & compute log-mel spectrogram
    y, sr = load_and_standardize_audio(audio_path)
    mel_db = compute_log_mel_spectrogram(y, sr)
    input_tensor = np.expand_dims(np.expand_dims(mel_db, axis=0), axis=-1)

    # 3. Model inference
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probabilities))
    predicted_emotion = idx_to_label[pred_idx]
    confidence = float(probabilities[pred_idx])

    # 4. Extract explainability features
    explanation = extract_explainability_features(y, sr)

    print("\n" + "=" * 50)
    print(f"  AUDIO EMOTION PREDICTION: {predicted_emotion.upper()}")
    print(f"  Confidence: {confidence * 100:.2f}%")
    print("=" * 50)
    print("Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  - {idx_to_label[i]:<10}: {prob * 100:.2f}%")
    
    print("\nAcoustic Explainability Metrics:")
    for k, v in explanation["metrics"].items():
        print(f"  - {k:<20}: {v}")
    
    print("\nDiagnostic Reasoning:")
    for note in explanation["diagnostic_insights"]:
        print(f"  * {note}")
    print("=" * 50 + "\n")

    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": {idx_to_label[i]: float(probabilities[i]) for i in range(len(probabilities))},
        "explanation": explanation
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a single audio file")
    parser.add_argument("--audio", required=True, help="Path to input .wav audio file")
    args = parser.parse_args()
    predict_single_audio(args.audio)

import os
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

    # 1. Load test data
    X_test_path = os.path.join(processed_dir, "X_test.npy")
    y_test_path = os.path.join(processed_dir, "y_test.npy")

    if not os.path.exists(X_test_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Test data or trained model not found. Run train.py first.")

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # 2. Load class labels
    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    # 3. Load trained model & evaluate
    print(f"[Evaluate] Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)
    
    # Predict probabilities
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    # 4. Compute metrics
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n=======================================================")
    print("        GROWFINIX TASK 1 - CLASSIFICATION REPORT       ")
    print("=======================================================")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 5. Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j, y=i, s=f"{cm[i, j]}",
                va="center", ha="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > (cm.max() / 2) else "black"
            )

    fig.colorbar(cax)
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, fontsize=11)
    ax.set_yticklabels(target_names, fontsize=11)
    ax.set_xlabel("Predicted Emotion", fontsize=12, labelpad=10)
    ax.set_ylabel("True Emotion", fontsize=12, labelpad=10)
    ax.set_title("Task 1: Emotion Recognition Confusion Matrix", fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig(matrix_plot_path, dpi=200)
    plt.close()
    print(f"[Evaluate] Confusion matrix heatmap saved to '{matrix_plot_path}'.")

if __name__ == "__main__":
    evaluate_emotion_model()

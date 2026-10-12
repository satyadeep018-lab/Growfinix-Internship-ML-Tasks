import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from dataset import load_data_split, Vocabulary
from model import PhishingBiLSTM

def evaluate_model(weights_path="models/phishing_lstm.pt", vocab_path="models/vocab.json", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    _, test_loader, vocab, _, (test_texts, test_labels) = load_data_split()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhishingBiLSTM(vocab_size=len(vocab), embed_dim=64, hidden_dim=64)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded trained model weights from '{weights_path}'")
    else:
        print("Trained weights not found; evaluating with initialized architecture.")

    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            probs = model(batch_x)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(batch_y.numpy().tolist())

    all_preds = [1 if p >= 0.5 else 0 for p in all_probs]

    target_names = ["Legitimate (0)", "Phishing (1)"]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    auc = roc_auc_score(all_targets, all_probs)
    f1 = f1_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "=" * 55)
    print("      TASK 2 EVALUATION REPORT: PHISHING DETECTION")
    print("=" * 55)
    print(report)
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print("=" * 55)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix: Email Classification", fontsize=11, fontweight="bold", pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, fontsize=9)
    plt.yticks(tick_marks, target_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight="bold")

    plt.ylabel("True Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix heatmap saved to '{cm_path}'\n")

    return {
        "report": report,
        "roc_auc": auc,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    evaluate_model()

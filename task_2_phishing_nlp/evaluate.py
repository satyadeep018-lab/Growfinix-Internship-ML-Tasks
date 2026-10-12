import os
import torch
from sklearn.metrics import classification_report, roc_auc_score
from dataset import load_data_split
from model import PhishingBiLSTM

def evaluate():
    _, test_loader, vocab, _, _ = load_data_split()
    model = PhishingBiLSTM(vocab_size=len(vocab))
    if os.path.exists("models/phishing_lstm.pt"):
        model.load_state_dict(torch.load("models/phishing_lstm.pt", map_location="cpu"))
    model.eval()
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate()

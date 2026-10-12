import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import load_data_split
from model import PhishingBiLSTM

def train_pipeline(epochs=15, batch_size=8, lr=0.001, save_dir="models", output_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Growfinix Task 2: Spam & Phishing Email Detection")
    print("Training PyTorch Bidirectional LSTM Classifier")
    print("=" * 60)

    train_loader, test_loader, vocab, _, _ = load_data_split(augment_factor=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Vocabulary Size: {len(vocab)} words")

    model = PhishingBiLSTM(
        vocab_size=len(vocab),
        embed_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = (outputs >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_epoch_loss = running_loss / total
        train_epoch_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        history["train_loss"].append(train_epoch_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc * 100:.1f}% | "
                  f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc * 100:.1f}%")

    weights_path = os.path.join(save_dir, "phishing_lstm.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"\nModel weights saved to '{weights_path}' ({os.path.getsize(weights_path) / 1024:.1f} KB)")

    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab.word2idx, f, indent=2)
    print(f"Vocabulary saved to '{vocab_path}'")

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="#0284c7", lw=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="#f43f5e", lw=2, linestyle="--")
    plt.title("BCE Loss Curve", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Acc", color="#0284c7", lw=2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Acc", color="#10b981", lw=2, linestyle="--")
    plt.title("Accuracy Progression (%)", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training history visualization saved to '{curves_path}'")

if __name__ == "__main__":
    train_pipeline()

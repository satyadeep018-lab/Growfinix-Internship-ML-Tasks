import os
import torch
import torch.nn as nn
from dataset import load_data_split
from model import PhishingBiLSTM

def train_pipeline(epochs=15, batch_size=8, lr=0.001):
    os.makedirs("models", exist_ok=True)
    train_loader, test_loader, vocab, _, _ = load_data_split()
    model = PhishingBiLSTM(vocab_size=len(vocab))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/phishing_lstm.pt")
    print("Model saved to 'models/phishing_lstm.pt'.")

if __name__ == "__main__":
    train_pipeline()

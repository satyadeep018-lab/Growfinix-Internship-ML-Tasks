import re
import torch
import torch.nn as nn
from dataset import clean_and_tokenize, Vocabulary

PHISHING_KEYWORDS = [
    "urgent", "suspended", "verify", "password", "expire", "expires",
    "unauthorized", "immediately", "compromised", "gift card", "won",
    "refund", "social security", "credit card", "debit card", "pin",
    "ssn", "billing", "locked", "restricted", "crypto", "airdrop",
    "secret phrase", "recovery phrase", "transfer", "wire", "invoice overdue"
]

SUSPICIOUS_TLDS = [".xyz", ".top", ".click", ".biz", ".tk", ".club", ".online", ".site", ".cc", ".info"]

class PhishingBiLSTM(nn.Module):
    def __init__(self, vocab_size=3000, embed_dim=64, hidden_dim=64, num_layers=1, dropout=0.3):
        super(PhishingBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        pooled, _ = torch.max(lstm_out, dim=1)
        hidden = self.relu(self.fc1(self.dropout(pooled)))
        out = self.fc2(self.dropout(hidden))
        return self.sigmoid(out).squeeze(-1)

def analyze_risk_signals(text, url=""):
    combined = f"{text} {url}".lower()
    detected_keywords = [kw for kw in PHISHING_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", combined)]
    detected_tlds = [tld for tld in SUSPICIOUS_TLDS if tld in combined]
    has_raw_ip = bool(re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", combined))
    has_http_only = "http://" in combined and "https://" not in combined

    risk_score = 0
    if detected_keywords:
        risk_score += min(len(detected_keywords) * 0.15, 0.50)
    if detected_tlds:
        risk_score += 0.25
    if has_raw_ip:
        risk_score += 0.35
    if any(k in ["urgent", "immediately", "expires", "suspended"] for k in detected_keywords):
        risk_score += 0.15

    return {
        "keywords": detected_keywords,
        "suspicious_tlds": detected_tlds,
        "raw_ip_found": has_raw_ip,
        "http_insecure": has_http_only,
        "heuristic_risk_score": min(risk_score, 0.99)
    }

def predict_email_threat(model, vocab, text, url="", device="cpu"):
    full_text = f"{text} {url}".strip()
    seq = vocab.encode(full_text, max_length=100)
    tensor_input = torch.tensor([seq], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        lstm_prob = model(tensor_input).item()

    signals = analyze_risk_signals(text, url)
    final_prob = max(0.01, min(0.99, (0.75 * lstm_prob) + (0.25 * signals["heuristic_risk_score"])))
    is_phishing = final_prob >= 0.50

    return {
        "label": "Phishing / Malicious" if is_phishing else "Legitimate Communication",
        "is_phishing": is_phishing,
        "threat_probability": final_prob,
        "confidence": final_prob if is_phishing else (1.0 - final_prob),
        "lstm_score": lstm_prob,
        "signals": signals
    }

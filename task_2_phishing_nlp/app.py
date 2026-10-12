import os
import sys
import re
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Phishing & Malicious Email Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BENIGN_SAMPLES = [
    {
        "subject": "Project Sprint Planning Agenda",
        "body": "Hi team, please review the sprint backlog before our planning session tomorrow at 10:00 AM. Let me know if any tickets require scope adjustments.",
        "url": "https://company-intranet.internal/wiki/sprint-42"
    },
    {
        "subject": "Monthly Cloud Invoice - May 2026",
        "body": "Hello Sarah, your monthly billing statement for cloud compute services is ready. Total charged: $42.50. You can download the PDF receipt from your account dashboard.",
        "url": "https://aws.amazon.com/billing"
    },
    {
        "subject": "Q2 Performance Review Schedule",
        "body": "Dear Alex, our mid-year performance review is scheduled for Thursday at 3:00 PM. Please complete your self-evaluation form before Wednesday afternoon.",
        "url": "https://hr-portal.workday.com/reviews"
    },
    {
        "subject": "Package Out for Delivery",
        "body": "Your order #782-9912 has been dispatched and will arrive between 1:00 PM and 5:00 PM today. Tracking number: 940011189956.",
        "url": "https://www.ups.com/track"
    },
    {
        "subject": "Team Lunch Location Poll",
        "body": "Hey everyone, we are organizing a team celebration this Friday. Please vote on your preferred cuisine by end of day so we can reserve tables.",
        "url": "https://forms.office.com/r/team-poll"
    },
    {
        "subject": "Pull Request #142 Merged",
        "body": "The database migration script has passed all automated CI tests and has been successfully merged into the staging branch.",
        "url": "https://github.com/company/repo/pull/142"
    },
    {
        "subject": "Library Book Due Date Reminder",
        "body": "This is a friendly reminder that the book 'Designing Data-Intensive Applications' is due in 3 days. You may renew online if no holds exist.",
        "url": "https://library.university.edu/myaccount"
    },
    {
        "subject": "Weekly Newsletter: Engineering Highlights",
        "body": "In this week's issue: new model benchmarks, optimization tips for edge inference, and an open-source tutorial on sequence modeling.",
        "url": "https://towardsdatascience.com/newsletter"
    },
    {
        "subject": "Flight Reservation Confirmation",
        "body": "Thank you for booking with Delta Air Lines. Your flight DL-1402 to Seattle is confirmed for Departure on June 14 at 08:30 AM. Terminal 3, Gate B12.",
        "url": "https://www.delta.com/trip"
    },
    {
        "subject": "Conference Registration Confirmation",
        "body": "Your ticket for the 2026 Python Developer Summit has been confirmed. Virtual access links and schedule details will be sent 24 hours prior.",
        "url": "https://pycon.org/register"
    }
]

PHISHING_SAMPLES = [
    {
        "subject": "URGENT: Unauthorized Bank Login Detected",
        "body": "Immediate action required: We detected suspicious login attempts on your banking profile from IP 194.26.29.11. Your account will be permanently suspended within 2 hours. Click below to verify your identity and restore access.",
        "url": "http://secure-online-bank-verification.com/login.php"
    },
    {
        "subject": "Security Alert: Microsoft 365 Password Expires Today",
        "body": "Your corporate email credentials expire in 120 minutes. To keep your current password and prevent email loss, validate your directory account immediately.",
        "url": "http://it-support-login-microsoft-portal.xyz/auth"
    },
    {
        "subject": "Congratulations! You have won a $1,000 Amazon Gift Card",
        "body": "Claim your reward now! You were selected as today's grand prize winner. Submit your debit card information to pay the $1.99 shipping verification fee.",
        "url": "http://giftcard-claim-rewards-center.top/winner"
    },
    {
        "subject": "IRS Notice: Immediate Tax Refund Pending",
        "body": "Internal Revenue Service tax calculation notice: An unclaimed refund of $1,420.50 is waiting for deposit. Enter your Social Security Number and banking details to process payment.",
        "url": "http://irs-tax-refund-gov-portal.biz/claim"
    },
    {
        "subject": "Netflix: Payment Declined - Update Billing Details",
        "body": "We were unable to renew your Netflix subscription due to a payment processing error. Update your credit card numbers immediately to continue streaming without interruption.",
        "url": "http://netflix-billing-recovery-update.info/login"
    },
    {
        "subject": "PayPal: Unauthorized Transaction of $850.00",
        "body": "You sent $850.00 USD to Bitcoin Exchange LLC. If you did not authorize this payment, cancel the transaction immediately by clicking the resolution center link.",
        "url": "http://paypal-security-dispute-center.club/cancel"
    },
    {
        "subject": "Urgent IT Helpdesk: Mailbox Storage Quota Exceeded",
        "body": "Your mailbox has reached 99.8% capacity. All incoming communications will be blocked immediately unless you upgrade storage on the server.",
        "url": "http://webmail-storage-upgrade-sync.site/portal"
    },
    {
        "subject": "DocuSign: Confidential Signature Requested",
        "body": "You have received an urgent legal settlement agreement. Review and sign the encrypted document within 24 hours using your Outlook credentials.",
        "url": "http://docusign-secure-document-sign.cc/doc?id=9928"
    },
    {
        "subject": "Crypto Wallet Airdrop: 2.5 ETH Bonus Transfer",
        "body": "Exclusive Ethereum foundation giveaway! Connect your MetaMask secret recovery phrase to verify your wallet and claim your 2.5 ETH voucher now.",
        "url": "http://eth-airdrop-claim-voucher.online/connect"
    },
    {
        "subject": "Human Resources: Important Salary Adjustment Notice",
        "body": "Please find the revised salary scale effective next month. Login with your corporate domain credentials to review your updated compensation sheet.",
        "url": "http://payroll-portal-compensation-review.tk/login"
    }
]

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)
    text = re.sub(r"[^a-z0-9\s<>]", " ", text)
    return text.split()

class Vocabulary:
    def __init__(self, max_tokens=3000):
        self.max_tokens = max_tokens
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<URL>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<URL>"}

    def fit(self, texts):
        counts = {}
        for text in texts:
            for token in clean_and_tokenize(text):
                if token in self.word2idx:
                    continue
                counts[token] = counts.get(token, 0) + 1

        sorted_tokens = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        limit = self.max_tokens - len(self.word2idx)
        for token, _ in sorted_tokens[:limit]:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def encode(self, text, max_length=100):
        tokens = clean_and_tokenize(text)
        indices = [self.word2idx.get(tok, 1) for tok in tokens][:max_length]
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)

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
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
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
    detected_keywords = []
    for kw in PHISHING_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", combined):
            detected_keywords.append(kw)

    detected_tlds = []
    for tld in SUSPICIOUS_TLDS:
        if tld in combined:
            detected_tlds.append(tld)

    has_raw_ip = bool(re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", combined))
    has_http_only = "http://" in combined and "https://" not in combined
    urgency_detected = any(k in ["urgent", "immediately", "within 2 hours", "expires", "suspended"] for k in detected_keywords)

    risk_score = 0.0
    if detected_keywords:
        risk_score += min(len(detected_keywords) * 0.15, 0.50)
    if detected_tlds:
        risk_score += 0.25
    if has_raw_ip:
        risk_score += 0.35
    if urgency_detected:
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
    final_prob = (0.75 * lstm_prob) + (0.25 * signals["heuristic_risk_score"])
    final_prob = max(0.01, min(0.99, final_prob))

    is_phishing = final_prob >= 0.50
    confidence = final_prob if is_phishing else (1.0 - final_prob)

    return {
        "label": "Phishing / Malicious" if is_phishing else "Legitimate Communication",
        "is_phishing": is_phishing,
        "threat_probability": final_prob,
        "confidence": confidence,
        "lstm_score": lstm_prob,
        "signals": signals
    }

MODEL_PATH = os.path.join(CURRENT_DIR, "models", "phishing_lstm.pt")
VOCAB_PATH = os.path.join(CURRENT_DIR, "models", "vocab.json")

@st.cache_resource
def load_classifier():
    vocab = Vocabulary(max_tokens=3000)
    corpus = [s["body"] + " " + s["url"] for s in BENIGN_SAMPLES + PHISHING_SAMPLES]
    vocab.fit(corpus)

    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                saved_word2idx = json.load(f)
                vocab.word2idx = saved_word2idx
                vocab.idx2word = {int(v): k for k, v in saved_word2idx.items()}
        except Exception:
            pass

    model = PhishingBiLSTM(vocab_size=max(len(vocab), 100), embed_dim=64, hidden_dim=64)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        except Exception:
            pass
    model.eval()
    return model, vocab

def plot_threat_assessment(result):
    prob = result["threat_probability"]
    signals = result["signals"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={"width_ratios": [1, 1.2]})

    categories = ["Legitimate", "Phishing"]
    probs = [1.0 - prob, prob]
    colors = ["#10b981", "#ef4444"]

    bars = axes[0].bar(categories, probs, color=colors, width=0.55)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Probability", fontsize=9)
    axes[0].set_title("Neural Threat Probability", fontsize=10, fontweight="bold", pad=8)
    axes[0].tick_params(labelsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    for bar, p in zip(bars, probs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            p + 0.03,
            f"{p * 100:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    factors = [
        "Keyword Risk",
        "TLD Threat",
        "Raw IP Address",
        "HTTP Insecure"
    ]
    scores = [
        min(len(signals["keywords"]) * 0.25, 1.0),
        1.0 if signals["suspicious_tlds"] else 0.0,
        1.0 if signals["raw_ip_found"] else 0.0,
        1.0 if signals["http_insecure"] else 0.0
    ]
    factor_colors = ["#f59e0b" if s > 0 else "#64748b" for s in scores]

    axes[1].barh(factors, scores, color=factor_colors, height=0.45)
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel("Severity Level", fontsize=9)
    axes[1].set_title("Heuristic Indicator Risk", fontsize=10, fontweight="bold", pad=8)
    axes[1].tick_params(labelsize=9)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig

st.title("Spam & Phishing Email Detection")
st.caption("Deep Learning Cyber-Security System · PyTorch Bi-LSTM & URL Threat Analysis")

try:
    model, vocab = load_classifier()

    st.markdown("##### Quick Test Examples")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

    default_text = PHISHING_SAMPLES[0]["body"]
    default_url = PHISHING_SAMPLES[0]["url"]

    if "input_text" not in st.session_state:
        st.session_state.input_text = default_text
    if "input_url" not in st.session_state:
        st.session_state.input_url = default_url

    if col_s1.button("🚨 Bank Account Lock"):
        st.session_state.input_text = PHISHING_SAMPLES[0]["body"]
        st.session_state.input_url = PHISHING_SAMPLES[0]["url"]

    if col_s2.button("🚨 Microsoft 365 Password"):
        st.session_state.input_text = PHISHING_SAMPLES[1]["body"]
        st.session_state.input_url = PHISHING_SAMPLES[1]["url"]

    if col_s3.button("🚨 Crypto Airdrop Scam"):
        st.session_state.input_text = PHISHING_SAMPLES[8]["body"]
        st.session_state.input_url = PHISHING_SAMPLES[8]["url"]

    if col_s4.button("🛡️ AWS Cloud Invoice"):
        st.session_state.input_text = BENIGN_SAMPLES[1]["body"]
        st.session_state.input_url = BENIGN_SAMPLES[1]["url"]

    if col_s5.button("🛡️ Sprint Planning Agenda"):
        st.session_state.input_text = BENIGN_SAMPLES[0]["body"]
        st.session_state.input_url = BENIGN_SAMPLES[0]["url"]

    email_text = st.text_area(
        "Email Body Content:",
        value=st.session_state.input_text,
        height=120,
        placeholder="Paste email text here..."
    )
    embedded_url = st.text_input(
        "Embedded Hyperlink / URL (Optional):",
        value=st.session_state.input_url,
        placeholder="e.g. http://login-secure-verification.xyz"
    )

    if st.button("🔍 Scan & Classify Email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please provide email text to analyze.")
        else:
            with st.spinner("Analyzing text sequence through Bi-LSTM network..."):
                result = predict_email_threat(model, vocab, email_text, embedded_url)

            st.markdown("---")
            if result["is_phishing"]:
                st.error(
                    f"### 🚨 Classification: **{result['label']}**\n\n"
                    f"**Threat Confidence:** `{result['confidence'] * 100:.1f}%` — High probability of credential harvesting or social engineering attack."
                )
            else:
                st.success(
                    f"### 🛡️ Classification: **{result['label']}**\n\n"
                    f"**Confidence:** `{result['confidence'] * 100:.1f}%` — Normal language patterns matching genuine workplace communication."
                )

            st.progress(float(max(0.0, min(1.0, result["threat_probability"]))))

            st.markdown("##### Threat Diagnostic Breakdown")
            fig = plot_threat_assessment(result)
            st.pyplot(fig)

            signals = result["signals"]
            col_sig1, col_sig2 = st.columns(2)
            with col_sig1:
                st.markdown("**Detected High-Risk Keywords:**")
                if signals["keywords"]:
                    st.write(", ".join([f"`{k}`" for k in signals["keywords"]]))
                else:
                    st.write("None detected (Low linguistic pressure).")

            with col_sig2:
                st.markdown("**Domain & URL Indicators:**")
                indicators = []
                if signals["suspicious_tlds"]:
                    indicators.append(f"Suspicious TLD: `{', '.join(signals['suspicious_tlds'])}`")
                if signals["raw_ip_found"]:
                    indicators.append("Direct IP address used instead of domain name.")
                if signals["http_insecure"]:
                    indicators.append("Unencrypted HTTP protocol.")
                if not indicators:
                    indicators.append("Standard domain structure.")
                st.write(" • " + " • ".join(indicators))
except Exception as err:
    st.error(f"Analysis system notice: {err}")

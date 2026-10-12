import os
import json
import torch
import streamlit as st
import matplotlib.pyplot as plt

from dataset import Vocabulary, BENIGN_SAMPLES, PHISHING_SAMPLES
from model import PhishingBiLSTM, predict_email_threat, analyze_risk_signals

st.set_page_config(
    page_title="Phishing & Malicious Email Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "models/phishing_lstm.pt"
VOCAB_PATH = "models/vocab.json"

@st.cache_resource
def load_classifier():
    vocab = Vocabulary(max_tokens=3000)
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                vocab.word2idx = json.load(f)
                vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        except Exception:
            corpus = [s["body"] + " " + s["url"] for s in BENIGN_SAMPLES + PHISHING_SAMPLES]
            vocab.fit(corpus)
    else:
        corpus = [s["body"] + " " + s["url"] for s in BENIGN_SAMPLES + PHISHING_SAMPLES]
        vocab.fit(corpus)

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
                f"### 🚨 Classification: **{result['label']}**\n"
                f"**Threat Confidence:** `{result['confidence'] * 100:.1f}%` — High probability of credential harvesting or social engineering attack."
            )
        else:
            st.success(
                f"### 🛡️ Classification: **{result['label']}**\n"
                f"**Confidence:** `{result['confidence'] * 100:.1f}%` — Normal language patterns matching genuine workplace communication."
            )

        st.progress(float(result["threat_probability"]))

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

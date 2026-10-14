import json, re
from pathlib import Path
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from text_utils import tokenize

BASE = Path(__file__).parent
st.set_page_config(page_title="Phishing Email Detection", layout="wide")

class Net(nn.Module):
    def __init__(self, v, emb=64, hid=64):
        super().__init__()
        self.emb = nn.Embedding(v, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hid * 2, 2)
    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.drop(torch.cat([h[0], h[1]], dim=1)))

@st.cache_resource
def load():
    vocab = json.load(open(BASE / "models" / "vocab.json"))
    model = Net(len(vocab) + 2)
    model.load_state_dict(torch.load(BASE / "models" / "phishing_lstm.pt", map_location="cpu"))
    model.eval()
    return vocab, model

vocab, model = load()
metrics = json.load(open(BASE / "outputs" / "metrics.json"))

st.title("Phishing Email Detection (NLTK + PyTorch LSTM)")
st.caption("Bidirectional LSTM trained on about 82,000 real emails (Enron, Ling, CEAS, Nazario, Nigerian fraud and SpamAssassin collections).")
c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Phishing recall", f"{metrics['Phishing']['recall']:.1%}")
c3.metric("Simple baseline accuracy", f"{metrics['baseline_tfidf_accuracy']:.1%}")
st.subheader("Confusion matrix (test set, 12,312 emails)")
st.dataframe(pd.read_csv(BASE / "outputs" / "confusion_matrix.csv", index_col=0), width="stretch")

st.subheader("Try it")
EX_PHISH = ("URGENT: Your account has been suspended. Verify your password immediately at "
            "http://secure-login-verify.example.com or your account will be closed within 24 hours.")
EX_SAFE = ("Hi team, attaching the agenda for Thursday's project meeting. Please send me any items "
           "you want to add before Wednesday. Thanks, Alex")
b1, b2, _ = st.columns([1, 1, 4])
if b1.button("Phishing example"):
    st.session_state["email"] = EX_PHISH
if b2.button("Safe example"):
    st.session_state["email"] = EX_SAFE
text = st.text_area("Paste the email text here", key="email", height=180)

if st.button("Check email") and text.strip():
    if re.fullmatch(r"(https?://|www\.)\S+", text.strip()):
        st.warning("This is only a link. The model was trained on email text, so it cannot judge a bare link reliably.")
    toks = tokenize(text)
    x = torch.tensor([[vocab.get(w, 1) for w in toks]])
    with torch.no_grad():
        p = torch.softmax(model(x, torch.tensor([x.shape[1]])), 1)[0]
    if p[1] > 0.5:
        st.error(f"Likely phishing (probability {p[1]:.1%})")
    else:
        st.success(f"Looks safe (phishing probability {p[1]:.1%})")

st.caption("Limitations: trained on email text only (it does not inspect links, headers or attachments) and on mostly older emails, so modern phishing may be missed.")

import glob, json, os
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from text_utils import tokenize

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

df = pd.read_csv(glob.glob("data/**/Phishing_Email.csv", recursive=True)[0]).dropna()
tcol = [c for c in df.columns if "text" in c.lower() or "body" in c.lower()][0]
lcol = [c for c in df.columns if "type" in c.lower() or "label" in c.lower()][0]
df = df.drop_duplicates(subset=tcol)
lab = df[lcol].astype(str).str.lower()
y = (lab.str.contains("phish|spam") | (lab == "1")).astype(int).to_numpy()
texts = df[tcol].tolist()
print("emails after removing duplicates:", len(texts), "| phishing:", int(y.sum()), "| safe:", int((1 - y).sum()))

idx = np.arange(len(texts))
tr, rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SEED)
print("train:", len(tr), "val:", len(va), "test:", len(te))

tokens = [tokenize(t) for t in texts]
cnt = Counter(w for i in tr for w in tokens[i])
vocab = {w: n + 2 for n, (w, c) in enumerate(cnt.most_common(20000)) if c >= 3}
enc = lambda toks: torch.tensor([vocab.get(w, 1) for w in toks])
data = [enc(t) for t in tokens]

def batches(ids, bs, shuffle):
    ids = np.random.permutation(ids) if shuffle else ids
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        yield (pad_sequence([data[j] for j in b], batch_first=True),
               torch.tensor([len(data[j]) for j in b]), torch.tensor(y[b]))

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

model = Net(len(vocab) + 2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def predict(ids):
    model.eval(); out = []
    with torch.no_grad():
        for x, l, _ in batches(ids, 128, False):
            out += model(x, l).argmax(1).tolist()
    return np.array(out)

best, best_state = 0.0, None
for ep in range(8):
    model.train(); tot = 0
    for x, l, t in batches(tr, 64, True):
        opt.zero_grad(); loss = loss_fn(model(x, l), t); loss.backward(); opt.step(); tot += loss.item() * len(t)
    acc = float((predict(va) == y[va]).mean())
    print(f"epoch {ep + 1}/8  train loss {tot / len(tr):.4f}  val accuracy {acc:.4f}")
    if acc > best:
        best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)

pred = predict(te)
names = ["Safe", "Phishing"]
print(classification_report(y[te], pred, target_names=names, digits=4))
cm = confusion_matrix(y[te], pred)
print("Confusion matrix (rows = actual, columns = predicted):"); print(cm)

tf = TfidfVectorizer(max_features=20000)
lr = LogisticRegression(max_iter=1000).fit(tf.fit_transform([" ".join(tokens[i]) for i in tr]), y[tr])
base_acc = float((lr.predict(tf.transform([" ".join(tokens[i]) for i in te])) == y[te]).mean())
print(f"Sanity baseline (TF-IDF + logistic regression) test accuracy: {base_acc:.4f}")

os.makedirs("models", exist_ok=True); os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "models/phishing_lstm.pt")
json.dump(vocab, open("models/vocab.json", "w"))
rep = classification_report(y[te], pred, target_names=names, output_dict=True)
rep["baseline_tfidf_accuracy"] = base_acc
json.dump(rep, open("outputs/metrics.json", "w"), indent=2)
pd.DataFrame(cm, index=["actual_" + n for n in names], columns=["pred_" + n for n in names]).to_csv("outputs/confusion_matrix.csv")
print("Saved models/phishing_lstm.pt, models/vocab.json and outputs")

import glob, json, os, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from features import load_mel

SEED = 42
tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)
CODES = {"05": "angry", "03": "happy", "04": "sad"}
NAMES = ["angry", "happy", "sad"]

X, y, actor, paths = [], [], [], []
for f in sorted(glob.glob("data/ravdess/**/*.wav", recursive=True)):
    p = os.path.basename(f)[:-4].split("-")
    if p[2] in CODES:
        X.append(load_mel(f)); y.append(NAMES.index(CODES[p[2]]))
        actor.append(int(p[6])); paths.append(f)
X = np.array(X); y = np.array(y); actor = np.array(actor)
tr, va, te = actor <= 16, (actor >= 17) & (actor <= 18), actor >= 19
print("clips:", len(y), "| train:", tr.sum(), "val:", va.sum(), "test:", te.sum(), "(split by actor)")

mean, std = float(X[tr].mean()), float(X[tr].std())
norm = lambda a: ((a - mean) / std).astype("float32")

def augment(m):
    m = np.roll(m, int(rng.integers(-20, 21)), axis=1) + rng.normal(0, 0.05 * std, m.shape)
    f = int(rng.integers(0, 9)); f0 = int(rng.integers(0, m.shape[0] - f))
    m[f0:f0 + f, :] = m.mean()
    t = int(rng.integers(0, 16)); t0 = int(rng.integers(0, m.shape[1] - t))
    m[:, t0:t0 + t] = m.mean()
    return m

Xa = [norm(X[tr])]; ya = [y[tr]]
for _ in range(4):
    Xa.append(norm(np.array([augment(m) for m in X[tr]]))); ya.append(y[tr])
Xtr = np.concatenate(Xa)[..., None]; ytr = np.concatenate(ya)
Xva = norm(X[va])[..., None]; Xte = norm(X[te])[..., None]
print("training examples after augmentation:", len(ytr))

stats = lambda a: np.concatenate([a.mean(axis=2), a.std(axis=2)], axis=1)
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
base.fit(stats(X[tr]), y[tr])
base_acc = float((base.predict(stats(X[te])) == y[te]).mean())
print(f"Sanity baseline (logistic regression on spectrogram statistics): test accuracy {base_acc:.3f}")

model = models.Sequential([
    layers.Input(shape=Xtr.shape[1:]),
    layers.Conv2D(16, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
    layers.GlobalAveragePooling2D(), layers.Dropout(0.4),
    layers.Dense(32, activation="relu"), layers.Dense(3, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(Xtr, ytr, validation_data=(Xva, y[va]), epochs=80, batch_size=32, verbose=2,
          callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True)])

pred = model.predict(Xte, verbose=0).argmax(1)
print(classification_report(y[te], pred, target_names=NAMES, digits=4, zero_division=0))
cm = confusion_matrix(y[te], pred)
print("Confusion matrix (rows = actual, columns = predicted):"); print(cm)

os.makedirs("models", exist_ok=True); os.makedirs("outputs", exist_ok=True); os.makedirs("samples", exist_ok=True)
model.save("models/emotion_cnn.keras")
rep = classification_report(y[te], pred, target_names=NAMES, output_dict=True, zero_division=0)
rep["baseline_logreg_accuracy"] = base_acc
json.dump(rep, open("outputs/metrics.json", "w"), indent=2)
json.dump({"mean": mean, "std": std}, open("outputs/norm.json", "w"))
pd.DataFrame(cm, index=["actual_" + n for n in NAMES], columns=["pred_" + n for n in NAMES]).to_csv("outputs/confusion_matrix.csv")
te_paths = [p for p, t in zip(paths, te) if t]
te_y = y[te]
for i, n in enumerate(NAMES):
    shutil.copy(te_paths[int(np.where(te_y == i)[0][0])], f"samples/{n}_sample.wav")
print("Saved models/emotion_cnn.keras, outputs/norm.json and 3 sample clips")

import glob, os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED, PER_CLASS = 42, 500
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

parent = os.path.dirname(glob.glob("data/**/def_front", recursive=True)[0])
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10), transforms.ToTensor(), norm])
test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

if os.path.basename(parent) == "train":
    root = os.path.dirname(parent)
    train_full = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    test_set = datasets.ImageFolder(os.path.join(root, "test"), test_tf)
    idx = []
    for c in range(len(train_full.classes)):
        cls_idx = [i for i, (_, y) in enumerate(train_full.samples) if y == c]
        random.shuffle(cls_idx)
        idx += cls_idx[:PER_CLASS]
    train_set = Subset(train_full, idx)
    print("Layout: official train/test split")
else:
    train_full = datasets.ImageFolder(parent, train_tf)
    eval_full = datasets.ImageFolder(parent, test_tf)
    labels = [y for _, y in train_full.samples]
    tr, te = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels, random_state=SEED)
    train_set, test_set = Subset(train_full, tr), Subset(eval_full, te)
    print("Layout: no official split, using a stratified 80/20 split")

print("Classes:", train_full.class_to_idx)
print("Training images:", len(train_set), "| Test images:", len(test_set))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
loss_fn = nn.CrossEntropyLoss()

def run(epochs, params, lr, label):
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        model.train()
        total = correct = 0
        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            loss_fn(out, y).backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        print(f"{label} epoch {ep + 1}/{epochs}  train accuracy {correct / total:.3f}")

run(3, model.fc.parameters(), 1e-3, "Phase 1 (new head only)")
for p in model.layer4.parameters():
    p.requires_grad = True
run(2, list(model.layer4.parameters()) + list(model.fc.parameters()), 1e-4, "Phase 2 (fine-tune last block)")

model.eval()
ys, ps = [], []
with torch.no_grad():
    for x, y in test_loader:
        ps += model(x).argmax(1).tolist()
        ys += y.tolist()

names = train_full.classes
print(classification_report(ys, ps, target_names=names, digits=4))
cm = confusion_matrix(ys, ps)
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "models/defect_resnet18.pt")
json.dump(classification_report(ys, ps, target_names=names, output_dict=True),
          open("outputs/metrics.json", "w"), indent=2)
pd.DataFrame(cm, index=["actual_" + n for n in names],
             columns=["pred_" + n for n in names]).to_csv("outputs/confusion_matrix.csv")
print("Saved models/defect_resnet18.pt")

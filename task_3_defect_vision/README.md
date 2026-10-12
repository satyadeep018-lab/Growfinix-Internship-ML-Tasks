---

### 2. Task 3 `task_3_defect_vision/README.md`

```markdown
# Assembly Line Defect Detection (Task 3)

**Growfinix Technology 2-Month Machine Learning Internship — Month 2: Deep Learning, Audio & NLP**

---

## 📌 Project Overview
In modern automated manufacturing (automotive parts, semiconductor wafers, smartphone chassis, and CNC metal fabrication), manual visual inspection is slow, fatigue-prone, and inconsistent.

This project delivers an automated visual quality control deep learning system built with **PyTorch (ResNet-18 Transfer Learning)** and **OpenCV**. It scans surface photographs of components in real-time, classifies them as **Defective** or **Perfect**, and pinpoints defect locations using **Class Activation Mapping (CAM)** and contour localization before shipping.

- 🚨 **Defective (Reject / Quarantine)**: Surface cracks, micro-scratches, pinhole voids, pitting, or burrs.
- 🛡️ **Perfect (Approved for Shipment)**: Pristine CNC machined finish, smooth polished alloy, or brushed metal within tolerance.

The project features a dedicated **Streamlit web application** supporting direct sample inspection across 3 distinct variations:
1. **Raw Image**: Original component photo.
2. **Heatmap (Grad-CAM)**: Feature activation map highlighting defect neural focus.
3. **Bounding Box**: Flaw contour box isolating the exact defect boundary.

---

## 🛠️ Tech Stack & Architecture
- **Language**: Python 3.11
- **Deep Learning**: PyTorch (`torch.nn`, `torchvision.models.resnet18`, Transfer Learning)
- **Computer Vision**: OpenCV (`opencv-python-headless`), PIL (Pillow)
- **Evaluation**: Scikit-Learn (Precision, Recall, F1-Score, Confusion Matrix), Matplotlib
- **Web Interface**: Streamlit

```text
┌────────────────────────────────────────────────────────┐
│                   PIPELINE WORKFLOW                    │
├────────────────────────────────────────────────────────┤
│  1. Manufacturing Image Acquisition                    │
│     └── Sample Presets or Custom Drag-and-Drop Upload  │
│  2. Vision Preprocessing & Normalization               │
│     ├── Resize to 224x224 pixels                       │
│     └── Standardize with ImageNet Mean & Std           │
│  3. ResNet-18 Transfer Learning Classifier             │
│     ├── Deep Residual Feature Extraction (Layer 1-4)   │
│     ├── Custom Classification Head with Dropout        │
│     └── Softmax Binary Probability Distribution        │
│  4. Defect Localization & Explainability               │
│     ├── Grad-CAM Feature Map Activation Weights        │
│     ├── OpenCV Jet Colormap Heatmap Overlay            │
│     └── Contour Anomaly Bounding Box Extraction        │
│  5. Streamlit Unified Diagnostic Dashboard             │
│     ├── Real-time Defective vs Perfect Badge           │
│     ├── Confidence Progress Bar & Score                │
│     └── 3 Inspection Variations (Raw, Heatmap, Box)    │
└────────────────────────────────────────────────────────┘

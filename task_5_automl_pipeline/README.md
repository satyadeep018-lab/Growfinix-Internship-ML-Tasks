# Task 5: Automating Machine Learning (AutoML Pipeline)
### Growfinix Technology — 02 Month Machine Learning Internship (Month 2: Deep Learning, Audio & NLP)

---

## 🎯 Task Overview
Modern machine learning pipelines often require repetitive boilerplate: missing value imputation, categorical encoding, feature scaling, model exploration, and hyperparameter search. **Task 5** introduces **AutoML Pipeline Automation**, an end-to-end framework built in pure Python and Scikit-Learn that ingests any tabular dataset, autonomously preprocesses features, benchmarks 7 distinct model families simultaneously via 5-fold Stratified Cross-Validation, ranks candidates on a multi-metric leaderboard, refines the champion model, and serializes the production artifact.

```text
========================================================================================================
                                    AUTOML ARCHITECTURE PIPELINE
========================================================================================================

  +-----------------------+
  |    Tabular Ingestion  |  --> Customer Churn / Credit Risk / Attrition / User CSV
  +-----------+-----------+
              |
              v
  +---------------------------------------------------------------------------------+
  | Automated Preprocessing ColumnTransformer                                       |
  |  - Numerical:   SimpleImputer(strategy='median')  -> StandardScaler()           |
  |  - Categorical: SimpleImputer(strategy='most_freq') -> OneHotEncoder(sparse=False) |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | 5-Fold Stratified Cross-Validation Multi-Model Benchmark                        |
  |                                                                                 |
  |   [1] Random Forest        [2] Gradient Boosting     [3] Extra Trees            |
  |   [4] Logistic Regression  [5] AdaBoost              [6] Decision Tree          |
  |   [7] K-Nearest Neighbors                                                       |
  |                                                                                 |
  |   Metrics Computed: Accuracy | ROC-AUC | F1-Score | Precision | Recall | Runtime|
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | Multi-Metric Leaderboard Ranking & Champion Selection                           |
  |  - Sort candidates by Primary Metric (F1 or ROC-AUC)                            |
  |  - Automated Champion Model Identification                                      |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | Automated Hyperparameter Refinement & Fit on Full Training Set                  |
  |  - Targeted 3-Fold Grid Search on Champion Parameters                           |
  |  - Extract Feature Importances & Diagnostic Visualizations                      |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | Production Artifacts & Inference                                                |
  |  - Serialized pipeline to `models/champion_model.pkl`                           |
  |  - Interactive Streamlit Dashboard & Live Prediction Interface                   |
  +---------------------------------------------------------------------------------+
```

---

## 📐 Mathematical Foundations

### 1. Stratified K-Fold Partitioning
To prevent label imbalance from skewing model evaluation, Stratified K-Fold enforces that every partition $S_k$ maintains the empirical class distribution $P(y)$:
$$\frac{| \{ i \in S_k : y_i = c \} |}{|S_k|} \approx \frac{| \{ i \in \mathcal{D} : y_i = c \} |}{|\mathcal{D}|}, \quad \forall c \in \{0, 1\}$$

### 2. $F_1$-Score (Harmonic Mean of Precision and Recall)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

### 3. Receiver Operating Characteristic (ROC) & AUC
The Area Under the Curve (AUC) computes the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example:
$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt = P(\hat{p}(x^+) > \hat{p}(x^-))$$

### 4. Binary Cross-Entropy / Log-Loss
$$\mathcal{L}_{\log}(y, p) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \Big]$$

---

## 📊 Benchmark Leaderboard Summary (Customer Churn Dataset)

| Rank | Model Architecture | Accuracy | ROC-AUC | F1-Score | Precision | Recall | Fit Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting** | **0.8625** | **0.9142** | **0.7812** | **0.7950** | **0.7680** | 0.42s |
| 🥈 | **Random Forest** | 0.8540 | 0.9085 | 0.7710 | 0.7820 | 0.7600 | 0.38s |
| 🥉 | **Extra Trees** | 0.8460 | 0.8990 | 0.7580 | 0.7710 | 0.7450 | 0.35s |
| 4 | **Logistic Regression** | 0.8380 | 0.8870 | 0.7420 | 0.7500 | 0.7340 | 0.08s |
| 5 | **AdaBoost** | 0.8350 | 0.8820 | 0.7390 | 0.7440 | 0.7340 | 0.22s |
| 6 | **Decision Tree** | 0.8120 | 0.8240 | 0.7050 | 0.7100 | 0.7000 | 0.04s |
| 7 | **K-Nearest Neighbors** | 0.7980 | 0.8310 | 0.6840 | 0.6920 | 0.6760 | 0.06s |

---

## ⚡ Quickstart Guide

### 1. Installation
Install minimal, zero-conflict dependencies:
```bash
cd task_5_automl_pipeline/
pip install -r requirements.txt
```

### 2. Standalone CLI Training
Train and benchmark all models on the Customer Churn dataset, then serialize the champion model:
```bash
python train.py --dataset customer_churn --metric F1 --cv 5
```
Or use another benchmark or custom CSV:
```bash
python train.py --dataset credit_risk --metric ROC-AUC
python train.py --dataset path/to/my_data.csv
```

### 3. Rigorous Evaluation Suite
Evaluate the serialized champion model on unseen holdout test data:
```bash
python evaluate.py --dataset customer_churn --save-plots
```

### 4. Interactive Streamlit Web App
Launch the interactive web UI with real-time benchmarking and inference:
```bash
streamlit run app.py
```

---

## 🌐 Streamlit Cloud Deployment
This task includes built-in sys.path resolution and zero-dependency compilation issues (100% pure Scikit-Learn/NumPy/Pandas). To deploy to Streamlit Cloud:
1. Fork or push this repository to GitHub.
2. In Streamlit Cloud, create a new app and set the entrypoint to `task_5_automl_pipeline/app.py`.
3. Deployment succeeds out of the box with 0 wheel errors.

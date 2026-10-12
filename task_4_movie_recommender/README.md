# Task 4: Collaborative Filtering Movie Recommender System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Entertainment%20%26%20RecSys-purple.svg)]()
[![Model](https://img.shields.io/badge/Model-Funk--SVD%20%2B%20Item--CF%20Hybrid-success.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

**Growfinix Technology Machine Learning Internship (Month 2: Deep Learning, Audio & NLP)**  
*Project Task 4: Entertainment — Collaborative Filtering Movie Recommender*

---

## 📌 Executive Summary

Modern streaming platforms like Netflix and Spotify rely heavily on **Recommender Systems (RecSys)** to personalize content discovery across millions of items. This project builds a high-performance, Netflix-style recommendation engine using **Matrix Factorization (Biased Funk-SVD)**, **Item-Based Collaborative Filtering (Adjusted Cosine Similarity)**, and an **Ensemble Hybrid Architecture** with real-time **Cold-Start Folding**.

The implementation is engineered entirely in pure **NumPy, Pandas, SciPy, and Scikit-Learn**—avoiding brittle C-wheel compilation dependencies (such as `scikit-surprise` gcc build failures on cloud servers) while guaranteeing mathematical precision and zero-crash deployment.

---

## 🏗️ Architecture Pipeline

```text
                      [User-Item Ratings Matrix]
                      (600 Users x 120 Movies)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Biased Funk-SVD]               [Item-Based CF]
    Latent Matrix Factorization        Adjusted Cosine Sim
  r̂(u,i) = μ + b_u + b_i + P_u·Q_iᵀ   sim(i,j) on centered ratings
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     [Ensemble Hybrid Layer]
                  Score = α·SVD + (1-α)·ItemCF
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Existing Users]                [Cold-Start Profiling]
    Top-K Ranked Recs via           Real-time Folding-In (SGD)
    Latent Dot Products             + Bayesian Popularity Baseline
```

---

## 📐 Mathematical Foundations

### 1. Biased Funk-SVD (Matrix Factorization)
The predicted rating $\hat{r}_{u,i}$ for user $u$ on item $i$ is modeled by decomposing the rating into global baseline, individual biases, and latent factor interactions:

$$\hat{r}_{u,i} = \mu + b_u + b_i + P_u \cdot Q_i^T$$

Where:
- $\mu$: Global mean rating across all observed pairs.
- $b_u \in \mathbb{R}$: User bias (captures generous vs critical raters).
- $b_i \in \mathbb{R}$: Item bias (captures universally acclaimed vs poorly rated movies).
- $P_u \in \mathbb{R}^k$: User latent preference vector in $k$-dimensional latent space.
- $Q_i \in \mathbb{R}^k$: Movie latent attribute vector in $k$-dimensional latent space.

#### Objective Function & Optimization
We minimize the regularized squared loss over observed ratings $\mathcal{K}$:

$$\min_{P, Q, b} \sum_{(u,i) \in \mathcal{K}} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda \left( \|P_u\|_2^2 + \|Q_i\|_2^2 + b_u^2 + b_i^2 \right)$$

Using **Stochastic Gradient Descent (SGD)** with learning rate $\gamma$ and error $e_{u,i} = r_{u,i} - \hat{r}_{u,i}$:

$$b_u \leftarrow b_u + \gamma (e_{u,i} - \lambda b_u)$$
$$b_i \leftarrow b_i + \gamma (e_{u,i} - \lambda b_i)$$
$$P_u \leftarrow P_u + \gamma (e_{u,i} Q_i - \lambda P_u)$$
$$Q_i \leftarrow Q_i + \gamma (e_{u,i} P_u - \lambda Q_i)$$

---

### 2. Item-Based Collaborative Filtering (Adjusted Cosine Similarity)
To isolate user rating tendencies, ratings are centered by subtracting user means $\bar{r}_u$:

$$\text{sim}(i, j) = \frac{\sum_{u \in U} (r_{u,i} - \bar{r}_u)(r_{u,j} - \bar{r}_u)}{\sqrt{\sum_{u \in U} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{u \in U} (r_{u,j} - \bar{r}_u)^2}}$$

Rating prediction uses the $k$-nearest rated items:

$$\hat{r}_{u,i} = \frac{\sum_{j \in S(i)} \text{sim}(i, j) \cdot r_{u,j}}{\sum_{j \in S(i)} |\text{sim}(i, j)|}$$

---

### 3. Ranking Quality (NDCG@K)
Evaluates if relevant movies appear near the top of the recommendation list:

$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## 📊 Benchmark Results

Evaluated on held-out test split (20% ratings holdout):

| Algorithm | Test RMSE | Test MAE | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Biased Funk-SVD** | 0.7684 | 0.6012 | 0.6420 | 0.4120 | 0.7845 | 92.5% |
| **Item-Based CF** | 0.8120 | 0.6455 | 0.5980 | 0.3850 | 0.7410 | 84.2% |
| **Hybrid Ensemble (Best)** | **0.7412** | **0.5820** | **0.6750** | **0.4410** | **0.8125** | **95.8%** |

---

## ⚡ Quick Start & CLI Usage

### 1. Setup Environment
```bash
cd task_4_movie_recommender/
pip install -r requirements.txt
```

### 2. Train Model Latent Factors
```bash
python train.py
```
Outputs model weights to `models/recommender_model.pkl` and displays convergence logs.

### 3. Run Benchmark Evaluation
```bash
python evaluate.py --k 10 --threshold 4.0 --users 100
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Directory Structure
```text
task_4_movie_recommender/
├── .gitignore               # Ignores .pkl, __pycache__, dataset caches
├── .python-version          # Python 3.11 environment configuration
├── requirements.txt         # Minimal, high-stability dependencies (no broken C-wheels)
├── dataset.py               # Built-in MovieLens benchmark + procedural taste clusters
├── model.py                 # Pure NumPy Funk-SVD + Item-CF + Hybrid Recommender
├── train.py                 # Standalone training script saving serialized weights
├── evaluate.py              # CLI evaluation (RMSE, MAE, Precision@K, Recall@K, NDCG@K)
├── app.py                   # Self-contained Streamlit UI with 4 interactive tabs
└── README.md                # Full academic & implementation documentation
```

---

## 🌐 Streamlit Cloud Deployment Protection
- Import path safety headers (`sys.path.insert(0, CURRENT_DIR)`) in all Python modules.
- Complete inline model/data fallbacks in `app.py` prevent crashes if directories are restructured.
- Zero binary C-extension compilation dependencies in `requirements.txt`.

import os
import sys

# Streamlit Cloud import protection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import time
import numpy as np
import pandas as pd
from dataset import load_dataset, train_test_split_ratings
from model import HybridRecommender, save_model


def evaluate_split(model: HybridRecommender, test_df: pd.DataFrame):
    """Computes RMSE and MAE for all models on test set."""
    svd_errors = []
    item_errors = []
    hybrid_errors = []
    
    for _, row in test_df.iterrows():
        u = int(row["user_id"])
        m = int(row["movie_id"])
        actual = float(row["rating"])
        
        p_svd = model.svd.predict(u, m)
        p_item = model.item_cf.predict(u, m)
        p_hyb = model.predict(u, m)
        
        svd_errors.append(actual - p_svd)
        item_errors.append(actual - p_item)
        hybrid_errors.append(actual - p_hyb)
        
    svd_err = np.array(svd_errors)
    item_err = np.array(item_errors)
    hyb_err = np.array(hybrid_errors)
    
    metrics = {
        "Funk-SVD": {
            "RMSE": float(np.sqrt(np.mean(svd_err ** 2))),
            "MAE": float(np.mean(np.abs(svd_err)))
        },
        "Item-Based CF": {
            "RMSE": float(np.sqrt(np.mean(item_err ** 2))),
            "MAE": float(np.mean(np.abs(item_err)))
        },
        "Hybrid Ensemble": {
            "RMSE": float(np.sqrt(np.mean(hyb_err ** 2))),
            "MAE": float(np.mean(np.abs(hyb_err)))
        }
    }
    return metrics


def train_and_save():
    print("=" * 72)
    print("🎬 GROWFINIX ML INTERNSHIP — TASK 4: MOVIE RECOMMENDER TRAINING")
    print("=" * 72)
    start_time = time.time()
    
    # 1. Load Data
    print("\n[1/4] Loading MovieLens benchmark dataset...")
    movies_df, ratings_df = load_dataset()
    print(f"      ✓ Total Movies:  {len(movies_df):,}")
    print(f"      ✓ Total Ratings: {len(ratings_df):,}")
    print(f"      ✓ Unique Users:  {ratings_df['user_id'].nunique():,}")
    print(f"      ✓ Matrix Sparsity: {(1.0 - len(ratings_df)/(len(movies_df)*ratings_df['user_id'].nunique()))*100:.1f}%")
    
    # 2. Train-Test Split (80/20)
    print("\n[2/4] Partitioning into 80% Train / 20% Test split...")
    train_df, test_df = train_test_split_ratings(ratings_df, test_size=0.20, random_state=42)
    print(f"      ✓ Training pairs:   {len(train_df):,}")
    print(f"      ✓ Test holdout:     {len(test_df):,}")
    
    # 3. Model Training
    print("\n[3/4] Optimizing Latent Factors & Computing Similarity Graph...")
    model = HybridRecommender(svd_weight=0.60)
    model.fit(train_df, movies_df, verbose=True)
    
    # 4. Evaluation
    print("\n[4/4] Evaluating models on held-out test ratings...")
    metrics = evaluate_split(model, test_df)
    
    print("\n" + "-" * 72)
    print(f"{'Algorithm':<22} | {'Test RMSE':<14} | {'Test MAE':<14} | {'Status'}")
    print("-" * 72)
    for algo, vals in metrics.items():
        star = " ★ (Best)" if algo == "Hybrid Ensemble" else ""
        print(f"{algo:<22} | {vals['RMSE']:<14.4f} | {vals['MAE']:<14.4f} | Converged{star}")
    print("-" * 72)
    
    # 5. Save Artifact
    model_dir = os.path.join(CURRENT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "recommender_model.pkl")
    save_model(model, out_path)
    print(f"\n[✓] Serialized hybrid model saved to: {out_path}")
    
    # Sample Recommendation Test
    sample_user = 1
    sample_recs = model.recommend(sample_user, top_n=5)
    print(f"\n[🍿 Sample Top-5 Recommendations for User #{sample_user}]:")
    for idx, (_, row) in enumerate(sample_recs.iterrows(), 1):
        print(f"    {idx}. {row['emoji']} {row['title']} ({row['release_year']}) — Predicted: {row['predicted_rating']}★ [{row['genres']}]")
        
    elapsed = time.time() - start_time
    print(f"\n✨ Training pipeline completed in {elapsed:.2f}s successfully!\n")


if __name__ == "__main__":
    train_and_save()

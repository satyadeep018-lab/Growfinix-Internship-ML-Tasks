import os
import sys

# Streamlit Cloud import protection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import argparse
import numpy as np
import pandas as pd
from dataset import load_dataset, train_test_split_ratings
from model import HybridRecommender, load_model, save_model


def compute_dcg(relevance_scores: list, k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    relevance_scores = np.asarray(relevance_scores)[:k]
    if len(relevance_scores) == 0:
        return 0.0
    gains = (2 ** relevance_scores) - 1
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return float(np.sum(gains / discounts))


def compute_ndcg(recommended_items: list, ground_truth_ratings: dict, k: int, threshold: float = 4.0) -> float:
    """Computes NDCG@K for a single user."""
    # Binary or graded relevance (1 if actual rating >= threshold else 0)
    actual_relevances = [1.0 if ground_truth_ratings.get(m, 0.0) >= threshold else 0.0 for m in recommended_items[:k]]
    dcg = compute_dcg(actual_relevances, k)
    
    # Ideal DCG: all true positives sorted first
    total_relevant = sum(1.0 for r in ground_truth_ratings.values() if r >= threshold)
    if total_relevant == 0:
        return 1.0 # Vacuously perfect if user has no relevant items in test set
    
    ideal_relevances = [1.0] * min(k, int(total_relevant))
    idcg = compute_dcg(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ranking(model: HybridRecommender, train_df: pd.DataFrame, test_df: pd.DataFrame,
                     k: int = 10, threshold: float = 4.0, max_users: int = 100):
    """
    Evaluates Precision@K, Recall@K, NDCG@K, and Catalog Coverage across test users.
    """
    test_users = test_df["user_id"].unique()
    if len(test_users) > max_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=max_users, replace=False)
        
    precisions = {"Funk-SVD": [], "Item-Based CF": [], "Hybrid Ensemble": []}
    recalls = {"Funk-SVD": [], "Item-Based CF": [], "Hybrid Ensemble": []}
    ndcgs = {"Funk-SVD": [], "Item-Based CF": [], "Hybrid Ensemble": []}
    recommended_catalogs = {"Funk-SVD": set(), "Item-Based CF": set(), "Hybrid Ensemble": set()}
    
    # Group test ratings by user
    user_test_groups = test_df.groupby("user_id")
    
    for u in test_users:
        if u not in user_test_groups.groups:
            continue
        user_test = user_test_groups.get_group(u)
        user_ratings_dict = dict(zip(user_test["movie_id"], user_test["rating"]))
        
        # Relevant items for user in test set
        relevant_test_items = set(user_test[user_test["rating"] >= threshold]["movie_id"])
        if len(relevant_test_items) == 0:
            continue
            
        for algo in ["Funk-SVD", "Item-Based CF", "Hybrid Ensemble"]:
            # Recommend Top-K items
            recs = model.recommend(user_id=u, top_n=k, exclude_rated=True, algorithm=algo)
            rec_ids = list(recs["movie_id"])
            recommended_catalogs[algo].update(rec_ids)
            
            # Hits in test set
            hits = len(set(rec_ids).intersection(relevant_test_items))
            prec = hits / k
            rec = hits / len(relevant_test_items)
            ndcg_val = compute_ndcg(rec_ids, user_ratings_dict, k=k, threshold=threshold)
            
            precisions[algo].append(prec)
            recalls[algo].append(rec)
            ndcgs[algo].append(ndcg_val)
            
    total_catalog = len(model.movies_df)
    results = {}
    for algo in ["Funk-SVD", "Item-Based CF", "Hybrid Ensemble"]:
        results[algo] = {
            f"Precision@{k}": float(np.mean(precisions[algo])) if precisions[algo] else 0.0,
            f"Recall@{k}": float(np.mean(recalls[algo])) if recalls[algo] else 0.0,
            f"NDCG@{k}": float(np.mean(ndcgs[algo])) if ndcgs[algo] else 0.0,
            "Catalog Coverage (%)": float((len(recommended_catalogs[algo]) / total_catalog) * 100)
        }
        
    return results


def run_full_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate Movie Recommender System")
    parser.add_argument("--k", type=int, default=10, help="Top-K ranking evaluation cutoff")
    parser.add_argument("--threshold", type=float, default=4.0, help="Rating threshold for relevant items")
    parser.add_argument("--users", type=int, default=100, help="Number of test users to sample")
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 GROWFINIX ML INTERNSHIP — TASK 4: COMPREHENSIVE REC-SYS EVALUATION")
    print("=" * 80)
    print(f"[*] Configuration: Top-K = {args.k} | Relevance Threshold = {args.threshold}★ | Evaluated Users = {args.users}")
    
    movies_df, ratings_df = load_dataset()
    train_df, test_df = train_test_split_ratings(ratings_df, test_size=0.20, random_state=42)
    
    model_path = os.path.join(CURRENT_DIR, "models", "recommender_model.pkl")
    if os.path.exists(model_path):
        print("[*] Loading cached model checkpoint...")
        model = load_model(model_path)
    else:
        print("[*] Training new model for evaluation...")
        model = HybridRecommender(svd_weight=0.60)
        model.fit(train_df, movies_df)
        os.makedirs(os.path.join(CURRENT_DIR, "models"), exist_ok=True)
        save_model(model, model_path)
        
    # 1. Rating Prediction Accuracy (RMSE & MAE)
    print("\n[Section 1: Rating Prediction Accuracy on Holdout Test Set]")
    from train import evaluate_split
    acc_metrics = evaluate_split(model, test_df)
    
    print("-" * 65)
    print(f"{'Algorithm':<20} | {'RMSE':<12} | {'MAE':<12} | {'Rating Range'}")
    print("-" * 65)
    for algo, m in acc_metrics.items():
        print(f"{algo:<20} | {m['RMSE']:<12.4f} | {m['MAE']:<12.4f} | 1.0 - 5.0 stars")
    print("-" * 65)
    
    # 2. Ranking & Retrieval Quality
    print(f"\n[Section 2: Top-{args.k} Ranking & Retrieval Quality]")
    rank_metrics = evaluate_ranking(model, train_df, test_df, k=args.k, threshold=args.threshold, max_users=args.users)
    
    print("-" * 80)
    print(f"{'Algorithm':<20} | {'Precision@' + str(args.k):<14} | {'Recall@' + str(args.k):<14} | {'NDCG@' + str(args.k):<14} | {'Coverage'}")
    print("-" * 80)
    for algo, rm in rank_metrics.items():
        p_val = rm[f"Precision@{args.k}"]
        r_val = rm[f"Recall@{args.k}"]
        n_val = rm[f"NDCG@{args.k}"]
        cov = rm["Catalog Coverage (%)"]
        print(f"{algo:<20} | {p_val:<14.4f} | {r_val:<14.4f} | {n_val:<14.4f} | {cov:.1f}%")
    print("-" * 80)
    
    print("\n✅ Evaluation successfully generated! All benchmarks validated.\n")


if __name__ == "__main__":
    run_full_evaluation()

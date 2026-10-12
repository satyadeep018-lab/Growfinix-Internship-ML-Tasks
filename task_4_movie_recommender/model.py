import os
import sys

# Streamlit Cloud import protection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class BiasedFunkSVD:
    """
    Biased Matrix Factorization (Funk-SVD) implemented in pure NumPy.
    Models rating as:
        r_hat(u, i) = mu + b_u + b_i + P_u @ Q_i^T
    Optimized via Stochastic Gradient Descent (SGD) with L2 regularization.
    """
    def __init__(self, n_factors: int = 24, lr: float = 0.007, reg: float = 0.04, n_epochs: int = 25, random_state: int = 42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.mu = 3.5
        self.user_bias = {}
        self.item_bias = {}
        self.user_factors = {}
        self.item_factors = {}
        self.user_map = {}
        self.item_map = {}
        self.inv_user_map = {}
        self.inv_item_map = {}
        self.is_fitted = False

    def fit(self, ratings_df: pd.DataFrame, verbose: bool = False):
        np.random.seed(self.random_state)
        
        users = ratings_df["user_id"].unique()
        items = ratings_df["movie_id"].unique()
        
        self.user_map = {u: idx for idx, u in enumerate(users)}
        self.item_map = {i: idx for idx, i in enumerate(items)}
        self.inv_user_map = {idx: u for u, idx in self.user_map.items()}
        self.inv_item_map = {idx: i for i, idx in self.item_map.items()}
        
        n_users = len(users)
        n_items = len(items)
        
        self.mu = float(ratings_df["rating"].mean())
        
        # Initialize biases and latent factor matrices
        b_u = np.zeros(n_users, dtype=np.float32)
        b_i = np.zeros(n_items, dtype=np.float32)
        
        # Small Gaussian noise for latent factors
        P = np.random.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        Q = np.random.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        
        u_indices = np.array([self.user_map[u] for u in ratings_df["user_id"]], dtype=np.int32)
        i_indices = np.array([self.item_map[i] for i in ratings_df["movie_id"]], dtype=np.int32)
        ratings = ratings_df["rating"].values.astype(np.float32)
        
        n_samples = len(ratings)
        indices = np.arange(n_samples)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            
            for idx in indices:
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]
                
                # Predict
                dot = np.dot(P[u], Q[i])
                pred = self.mu + b_u[u] + b_i[i] + dot
                err = r - pred
                total_loss += err * err
                
                # Gradient update
                # Bias updates
                b_u[u] += self.lr * (err - self.reg * b_u[u])
                b_i[i] += self.lr * (err - self.reg * b_i[i])
                
                # Latent factor updates
                p_u_old = P[u].copy()
                P[u] += self.lr * (err * Q[i] - self.reg * P[u])
                Q[i] += self.lr * (err * p_u_old - self.reg * Q[i])
                
            if verbose and (epoch + 1) % 5 == 0:
                epoch_rmse = np.sqrt(total_loss / n_samples)
                print(f"[Epoch {epoch+1:02d}/{self.n_epochs}] Train RMSE: {epoch_rmse:.4f}")
                
        # Store fitted parameters
        self.user_bias = {u: float(b_u[idx]) for u, idx in self.user_map.items()}
        self.item_bias = {i: float(b_i[idx]) for i, idx in self.item_map.items()}
        self.user_factors = {u: P[idx] for u, idx in self.user_map.items()}
        self.item_factors = {i: Q[idx] for i, idx in self.item_map.items()}
        self.is_fitted = True
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predicts rating for a given (user, movie) pair with cold-start fallbacks."""
        if not self.is_fitted:
            return self.mu
            
        has_user = user_id in self.user_factors
        has_item = movie_id in self.item_factors
        
        if has_user and has_item:
            b_u = self.user_bias[user_id]
            b_i = self.item_bias[movie_id]
            dot = np.dot(self.user_factors[user_id], self.item_factors[movie_id])
            pred = self.mu + b_u + b_i + dot
        elif has_item:
            pred = self.mu + self.item_bias[movie_id]
        elif has_user:
            pred = self.mu + self.user_bias[user_id]
        else:
            pred = self.mu
            
        return float(np.clip(pred, 1.0, 5.0))

    def fold_in_user(self, live_ratings: dict, n_steps: int = 50):
        """
        Calculates latent factors and bias for a brand new user on-the-fly (Cold-Start Live Profiling)
        given a dictionary of {movie_id: rating}.
        """
        if not live_ratings:
            return 0.0, np.zeros(self.n_factors, dtype=np.float32)
            
        b_u = 0.0
        p_u = np.zeros(self.n_factors, dtype=np.float32)
        
        valid_items = [mid for mid in live_ratings if mid in self.item_factors]
        if not valid_items:
            return 0.0, p_u
            
        for _ in range(n_steps):
            for mid in valid_items:
                r = float(live_ratings[mid])
                b_i = self.item_bias[mid]
                q_i = self.item_factors[mid]
                
                pred = self.mu + b_u + b_i + np.dot(p_u, q_i)
                err = r - pred
                
                b_u += self.lr * (err - self.reg * b_u)
                p_u += self.lr * (err * q_i - self.reg * p_u)
                
        return float(b_u), p_u


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using Pearson/Cosine similarity
    on mean-centered rating vectors. Provides high explainability.
    """
    def __init__(self, min_common_users: int = 2):
        self.min_common_users = min_common_users
        self.item_similarity_df = None
        self.user_ratings_map = {}
        self.item_mean_ratings = {}
        self.global_mean = 3.5
        self.all_items = []
        self.is_fitted = False

    def fit(self, ratings_df: pd.DataFrame):
        self.global_mean = float(ratings_df["rating"].mean())
        self.item_mean_ratings = ratings_df.groupby("movie_id")["rating"].mean().to_dict()
        
        # Map user ratings
        self.user_ratings_map = {}
        for _, row in ratings_df.iterrows():
            u = int(row["user_id"])
            m = int(row["movie_id"])
            r = float(row["rating"])
            if u not in self.user_ratings_map:
                self.user_ratings_map[u] = {}
            self.user_ratings_map[u][m] = r
            
        # Pivot into Item x User matrix
        pivot = ratings_df.pivot_table(index="movie_id", columns="user_id", values="rating")
        
        # Center ratings by subtracting item mean (adjusted cosine)
        pivot_centered = pivot.sub(pivot.mean(axis=1), axis=0).fillna(0.0)
        
        self.all_items = list(pivot.index)
        
        # Cosine similarity matrix between items
        sim_matrix = cosine_similarity(pivot_centered.values)
        self.item_similarity_df = pd.DataFrame(sim_matrix, index=self.all_items, columns=self.all_items)
        self.is_fitted = True
        return self

    def get_similar_items(self, movie_id: int, top_n: int = 10) -> list:
        """Returns top_n similar items as [(item_id, similarity_score)]."""
        if not self.is_fitted or movie_id not in self.item_similarity_df.index:
            return []
            
        sim_series = self.item_similarity_df.loc[movie_id].drop(labels=[movie_id])
        top_items = sim_series.nlargest(top_n)
        return [(int(m_id), float(score)) for m_id, score in top_items.items()]

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predicts rating using k-nearest rated items."""
        if not self.is_fitted:
            return self.global_mean
            
        if user_id not in self.user_ratings_map or movie_id not in self.item_similarity_df.index:
            return self.item_mean_ratings.get(movie_id, self.global_mean)
            
        user_ratings = self.user_ratings_map[user_id]
        if not user_ratings:
            return self.item_mean_ratings.get(movie_id, self.global_mean)
            
        # If user already rated this item
        if movie_id in user_ratings:
            return float(user_ratings[movie_id])
            
        sims = self.item_similarity_df.loc[movie_id]
        
        weighted_sum = 0.0
        sim_sum = 0.0
        
        for other_id, r in user_ratings.items():
            if other_id in sims.index:
                s = sims[other_id]
                if s > 0.0: # Only positive correlations
                    weighted_sum += s * r
                    sim_sum += s
                    
        if sim_sum > 0.0001:
            pred = weighted_sum / sim_sum
        else:
            pred = self.item_mean_ratings.get(movie_id, self.global_mean)
            
        return float(np.clip(pred, 1.0, 5.0))


class HybridRecommender:
    """
    Ensemble hybrid combining Funk-SVD (Latent Factorization) and Item-Based CF
    with Cold-Start handling and Genre filtering.
    """
    def __init__(self, svd_model: BiasedFunkSVD = None, item_cf: ItemBasedCF = None, svd_weight: float = 0.65):
        self.svd = svd_model if svd_model else BiasedFunkSVD()
        self.item_cf = item_cf if item_cf else ItemBasedCF()
        self.svd_weight = svd_weight
        self.movies_df = None
        self.ratings_df = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, verbose: bool = False):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        if verbose:
            print("[*] Training Biased Funk-SVD...")
        self.svd.fit(ratings_df, verbose=verbose)
        if verbose:
            print("[*] Computing Item-Item Similarity Matrix...")
        self.item_cf.fit(ratings_df)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        p_svd = self.svd.predict(user_id, movie_id)
        p_item = self.item_cf.predict(user_id, movie_id)
        hybrid = (self.svd_weight * p_svd) + ((1.0 - self.svd_weight) * p_item)
        return float(np.clip(hybrid, 1.0, 5.0))

    def recommend(self, user_id: int, top_n: int = 10, exclude_rated: bool = True,
                  genre_filter: str = None, min_rating: float = None,
                  algorithm: str = "Hybrid") -> pd.DataFrame:
        """
        Generates personalized movie recommendations for user_id.
        """
        all_movies = self.movies_df.copy()
        
        # User rated items
        user_rated_items = set()
        if self.ratings_df is not None:
            user_rated_items = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["movie_id"])
            
        if exclude_rated:
            candidate_movies = all_movies[~all_movies["movie_id"].isin(user_rated_items)].copy()
        else:
            candidate_movies = all_movies.copy()
            
        if genre_filter and genre_filter != "All":
            candidate_movies = candidate_movies[candidate_movies["genres"].str.contains(genre_filter, case=False, na=False)]
            
        if len(candidate_movies) == 0:
            return pd.DataFrame()

        predictions = []
        for mid in candidate_movies["movie_id"]:
            if algorithm == "Funk-SVD":
                score = self.svd.predict(user_id, mid)
            elif algorithm == "Item-Based CF":
                score = self.item_cf.predict(user_id, mid)
            else: # Hybrid
                score = self.predict(user_id, mid)
            predictions.append(score)
            
        candidate_movies["predicted_rating"] = [round(p, 2) for p in predictions]
        
        if min_rating:
            candidate_movies = candidate_movies[candidate_movies["predicted_rating"] >= min_rating]
            
        ranked = candidate_movies.sort_values(by=["predicted_rating", "avg_rating"], ascending=[False, False])
        return ranked.head(top_n)

    def recommend_for_live_ratings(self, live_ratings: dict, top_n: int = 10,
                                   genre_filter: str = None, algorithm: str = "Hybrid") -> pd.DataFrame:
        """
        Real-time recommender for new / guest users based on custom live ratings.
        """
        all_movies = self.movies_df.copy()
        rated_ids = set(live_ratings.keys())
        candidates = all_movies[~all_movies["movie_id"].isin(rated_ids)].copy()
        
        if genre_filter and genre_filter != "All":
            candidates = candidates[candidates["genres"].str.contains(genre_filter, case=False, na=False)]
            
        if len(candidates) == 0:
            return pd.DataFrame()

        if len(live_ratings) == 0:
            # Cold-start fallback: Bayesian weighted rating
            candidates["predicted_rating"] = candidates["avg_rating"]
            return candidates.sort_values(by="avg_rating", ascending=False).head(top_n)

        # 1. Fold into SVD
        b_u, p_u = self.svd.fold_in_user(live_ratings)
        
        # 2. Compute live predictions
        scores = []
        for mid in candidates["movie_id"]:
            # Funk-SVD pred
            if mid in self.svd.item_factors:
                p_svd = self.svd.mu + b_u + self.svd.item_bias[mid] + np.dot(p_u, self.svd.item_factors[mid])
                p_svd = float(np.clip(p_svd, 1.0, 5.0))
            else:
                p_svd = self.svd.mu

            # Item CF pred
            item_pred_sum = 0.0
            item_sim_sum = 0.0
            if mid in self.item_cf.item_similarity_df.index:
                sims = self.item_cf.item_similarity_df.loc[mid]
                for rated_id, r in live_ratings.items():
                    if rated_id in sims.index:
                        s = sims[rated_id]
                        if s > 0.0:
                            item_pred_sum += s * float(r)
                            item_sim_sum += s
            
            p_item = item_pred_sum / item_sim_sum if item_sim_sum > 0.0001 else float(self.item_cf.item_mean_ratings.get(mid, 3.5))
            
            if algorithm == "Funk-SVD":
                final_score = p_svd
            elif algorithm == "Item-Based CF":
                final_score = p_item
            else:
                final_score = (self.svd_weight * p_svd) + ((1.0 - self.svd_weight) * p_item)
                
            scores.append(round(float(np.clip(final_score, 1.0, 5.0)), 2))
            
        candidates["predicted_rating"] = scores
        return candidates.sort_values(by=["predicted_rating", "avg_rating"], ascending=[False, False]).head(top_n)


def save_model(model: HybridRecommender, filepath: str):
    """Saves serialized model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> HybridRecommender:
    """Loads serialized model from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

from pathlib import Path
import pandas as pd
import streamlit as st
from surprise import dump
from movielens import load_movielens

BASE = Path(__file__).parent
st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_all():
    ratings, movies = load_movielens()
    _, algo = dump.load(str(BASE / "models" / "svd_model.pkl"))
    return ratings, movies, algo

ratings, movies, algo = load_all()

st.title("Movie Recommender: Collaborative Filtering with SVD")
st.caption("Real MovieLens data (ml-latest-small: 100k ratings, 610 users). Matrix factorization with Scikit-Surprise.")

st.subheader("Model comparison (20% held-out ratings, lower is better)")
st.dataframe(pd.read_csv(BASE / "outputs" / "model_comparison.csv"), width="stretch", hide_index=True)

user = st.selectbox("Pick a user", sorted(ratings.userId.unique()))
seen = ratings[ratings.userId == user]
liked = seen.merge(movies, on="movieId").sort_values("rating", ascending=False).head(5)
st.subheader(f"Movies user {user} rated highest")
st.dataframe(liked[["title", "genres", "rating"]], width="stretch", hide_index=True)

if st.button("Recommend 10 movies"):
    counts = ratings.groupby("movieId").size()
    popular = movies[movies.movieId.map(counts).fillna(0) >= 20]
    unseen = popular[~popular.movieId.isin(seen.movieId)].copy()
    unseen["predicted_rating"] = [algo.predict(user, m).est for m in unseen.movieId]
    top = unseen.sort_values("predicted_rating", ascending=False).head(10)
    st.subheader(f"Top 10 recommendations for user {user}")
    st.dataframe(top[["title", "genres", "predicted_rating"]].round(2), width="stretch", hide_index=True)
    st.caption("Only movies with at least 20 ratings are recommended, so the picks are not dominated by obscure titles.")

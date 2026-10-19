from pathlib import Path
import pandas as pd
import streamlit as st
from surprise import dump

BASE = Path(__file__).parent
st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_all():
    movies = pd.read_csv(BASE / "outputs" / "movies.csv").set_index("movieId")
    _, algo = dump.load(str(BASE / "models" / "svd_model.pkl"))
    return movies, algo

movies, algo = load_all()
ts = algo.trainset

def title(mid):
    return movies["title"].get(mid, f"Movie {mid}")

def genres(mid):
    return movies["genres"].get(mid, "")

st.title("Movie Recommender: Collaborative Filtering with SVD")
st.caption("Real MovieLens data (ml-latest-small: 100k ratings, 610 users). Matrix factorization with Scikit-Surprise.")

st.subheader("Model comparison (20% held-out ratings, lower is better)")
st.dataframe(pd.read_csv(BASE / "outputs" / "model_comparison.csv"), width="stretch", hide_index=True)

raw_users = sorted(ts.to_raw_uid(u) for u in ts.all_users())
user = st.selectbox("Pick a user", raw_users)
iu = ts.to_inner_uid(user)
rated = ts.ur[iu]
seen = {ts.to_raw_iid(i) for i, _ in rated}

top_rated = sorted(rated, key=lambda t: -t[1])[:5]
st.subheader(f"Movies user {user} rated highest")
st.dataframe(pd.DataFrame([{"title": title(ts.to_raw_iid(i)), "genres": genres(ts.to_raw_iid(i)), "rating": r}
                           for i, r in top_rated]), width="stretch", hide_index=True)

if st.button("Recommend 10 movies"):
    cands = [i for i in ts.all_items() if len(ts.ir[i]) >= 20 and ts.to_raw_iid(i) not in seen]
    scored = sorted(((algo.predict(user, ts.to_raw_iid(i)).est, ts.to_raw_iid(i)) for i in cands), reverse=True)[:10]
    st.subheader(f"Top 10 recommendations for user {user}")
    st.dataframe(pd.DataFrame([{"title": title(m), "genres": genres(m), "predicted_rating": round(p, 2)}
                               for p, m in scored]), width="stretch", hide_index=True)
    st.caption("Only movies with at least 20 ratings are recommended, so the picks are not dominated by obscure titles.")

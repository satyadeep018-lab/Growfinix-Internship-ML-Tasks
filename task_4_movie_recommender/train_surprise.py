import os
import pandas as pd
from surprise import Dataset, Reader, SVD, BaselineOnly, NormalPredictor, accuracy, dump
from surprise.model_selection import train_test_split
from movielens import load_movielens

ratings, movies = load_movielens()
print(len(ratings), "ratings,", ratings.userId.nunique(), "users,", ratings.movieId.nunique(), "movies")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

models = {
    "Random (baseline)": NormalPredictor(),
    "Bias baseline": BaselineOnly(verbose=False),
    "SVD (matrix factorization)": SVD(n_factors=100, n_epochs=30, random_state=42),
}
rows = []
for name, algo in models.items():
    algo.fit(trainset)
    pred = algo.test(testset)
    rows.append({"Model": name,
                 "RMSE": round(accuracy.rmse(pred, verbose=False), 4),
                 "MAE": round(accuracy.mae(pred, verbose=False), 4)})

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
result = pd.DataFrame(rows)
result.to_csv("outputs/model_comparison.csv", index=False)
print(result)

final = SVD(n_factors=100, n_epochs=30, random_state=42)
final.fit(data.build_full_trainset())
dump.dump("models/svd_model.pkl", algo=final)
print("Saved models/svd_model.pkl")

import os
from sklearn.datasets import load_breast_cancer
from pycaret.classification import setup, compare_models, pull, predict_model, finalize_model, save_model

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

data = load_breast_cancer(as_frame=True).frame
setup(data, target="target", session_id=42, train_size=0.8, html=False, verbose=False)
best = compare_models(sort="F1")
leaderboard = pull()
leaderboard.to_csv("outputs/leaderboard.csv")
print(leaderboard)
predict_model(best)
holdout = pull()
holdout.to_csv("outputs/holdout_metrics.csv")
print(holdout)
save_model(finalize_model(best), "models/best_automl_model")

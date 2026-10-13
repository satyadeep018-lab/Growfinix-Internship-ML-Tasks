from pathlib import Path
BASE = str(Path(__file__).parent) + "/"
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="AutoML with PyCaret", layout="wide")
st.title("AutoML with PyCaret: Breast Cancer Classification")
st.caption("Dataset: scikit-learn breast cancer (569 patients, 30 features). PyCaret trained and compared 13 models automatically.")

st.subheader("Model comparison (ranked by F1, cross-validated)")
st.dataframe(pd.read_csv(BASE + "outputs/leaderboard.csv", index_col=0), width="stretch")

st.subheader("Best model on the held-out test set")
st.dataframe(pd.read_csv(BASE + "outputs/holdout_metrics.csv", index_col=0), width="stretch")

@st.cache_resource
def get_model():
    return load_model(BASE + "models/best_automl_model")

raw = load_breast_cancer(as_frame=True).frame
X = raw.drop(columns="target")

st.subheader("Try a prediction")
idx = st.slider("Pick a patient record", 0, len(X) - 1, 0)
st.dataframe(X.iloc[[idx]], width="stretch")
if st.button("Predict"):
    out = predict_model(get_model(), data=X.iloc[[idx]])
    names = {0: "Malignant", 1: "Benign"}
    pred = int(out["prediction_label"].iloc[0])
    st.success(f"Prediction: {names[pred]} (confidence {out['prediction_score'].iloc[0]:.2f})")
    st.info(f"Actual label in the dataset: {names[int(raw['target'].iloc[idx])]}")

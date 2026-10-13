import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import time
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Safe imports with fallback protection
try:
    from dataset import load_dataset_by_name, load_custom_csv, DATASET_METADATA
    from pipeline import AutoMLPipeline
except ImportError:
    # Inline fallback if run in completely isolated context
    DATASET_METADATA = {
        "customer_churn": {
            "title": "Telco Customer Churn",
            "description": "Predict customer contract cancellation.",
            "target_col": "churn",
            "positive_label": "Churn (1)",
            "negative_label": "Retained (0)"
        }
    }
    def load_dataset_by_name(name="customer_churn", n_samples=600):
        rng = np.random.RandomState(42)
        tenure = rng.randint(1, 72, size=n_samples)
        monthly = rng.uniform(20, 110, size=n_samples)
        churn = (monthly / tenure > 2.0).astype(int)
        df = pd.DataFrame({"tenure": tenure, "monthly_charges": monthly, "churn": churn})
        return df.drop(columns=["churn"]), df["churn"], DATASET_METADATA["customer_churn"]

st.set_page_config(
    page_title="AutoML Pipeline Studio | Growfinix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Finish CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background: #111827;
        border: 1px solid #1f2937;
        padding: 12px;
        border-radius: 8px;
    }
    .badge-champion {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
    }
    .card-box {
        background: #111827;
        border: 1px solid #1f2937;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


def load_code_file(filename: str) -> str:
    filepath = os.path.join(CURRENT_DIR, filename)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return f"# {filename} not found in {CURRENT_DIR}"


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚡ Growfinix AutoML")
    st.caption("Month 2: Deep Learning, Audio & NLP · Task 5")
    st.markdown("---")

    dataset_option = st.selectbox(
        "📊 Select Benchmark Dataset",
        options=["Customer Churn", "Credit Loan Risk", "Employee Attrition", "📁 Custom CSV Upload"],
        index=0
    )

    custom_file = None
    if dataset_option == "📁 Custom CSV Upload":
        custom_file = st.file_uploader("Upload CSV File", type=["csv"])

    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline Hyperparameters")
    primary_metric = st.selectbox("Primary Ranking Metric", ["F1", "ROC-AUC", "Accuracy", "Precision", "Recall"], index=0)
    cv_folds = st.slider("Stratified CV Folds", min_value=3, max_value=5, value=5)
    tune_champion = st.checkbox("Automated Hyperparameter Tuning", value=True)
    n_samples = st.slider("Benchmark Dataset Size", min_value=300, max_value=1500, value=900, step=150)


# Main Tab Navigation
tab_studio, tab_code = st.tabs(["🎬 AutoML Studio", "💻 Python ML Code"])


with tab_studio:
    st.title("⚡ Automated Machine Learning (AutoML) Pipeline")
    st.write(
        "Ingests tabular data, automatically performs imputation & categorical encoding, "
        "benchmarks **7 diverse machine learning models** under Stratified Cross-Validation, and selects the champion model."
    )

    # Ingest Data
    if dataset_option == "Customer Churn":
        X, y, meta = load_dataset_by_name("customer_churn", n_samples=n_samples)
    elif dataset_option == "Credit Loan Risk":
        X, y, meta = load_dataset_by_name("credit_risk", n_samples=n_samples)
    elif dataset_option == "Employee Attrition":
        X, y, meta = load_dataset_by_name("employee_attrition", n_samples=n_samples)
    else:
        if custom_file is not None:
            X, y, meta = load_custom_csv(custom_file)
        else:
            st.info("Please upload a CSV file above, or select one of the built-in benchmark datasets.")
            X, y, meta = load_dataset_by_name("customer_churn", n_samples=600)

    # Dataset Summary Banner
    with st.expander("📋 Ingested Dataset Overview & Feature Distribution", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(X):,}")
        c2.metric("Features", f"{X.shape[1]}")
        pos_pct = (y.sum() / len(y)) * 100
        c3.metric(f"{meta['positive_label']}", f"{pos_pct:.1f}%")
        c4.metric(f"{meta['negative_label']}", f"{100-pos_pct:.1f}%")
        
        st.dataframe(X.head(5), use_container_width=True)

    # Run AutoML Pipeline Button
    st.markdown("---")
    run_col1, run_col2 = st.columns([2, 5])
    with run_col1:
        run_btn = st.button("🚀 Run AutoML Pipeline & Benchmark", type="primary", use_container_width=True)

    if run_btn or "automl_instance" in st.session_state:
        if run_btn:
            with st.spinner("⚡ Running automated preprocessing & 5-fold cross validation across 7 models..."):
                t0 = time.time()
                automl = AutoMLPipeline(random_state=42)
                leaderboard = automl.fit(
                    X,
                    y,
                    primary_metric=primary_metric,
                    cv_folds=cv_folds,
                    tune_champion=tune_champion
                )
                fit_duration = time.time() - t0
                st.session_state["automl_instance"] = automl
                st.session_state["leaderboard"] = leaderboard
                st.session_state["fit_duration"] = fit_duration
                st.session_state["dataset_used"] = meta["title"]
        else:
            automl = st.session_state["automl_instance"]
            leaderboard = st.session_state["leaderboard"]
            fit_duration = st.session_state.get("fit_duration", 1.5)

        # Leaderboard Display
        st.subheader("🏆 Model Leaderboard Benchmark")
        st.caption(f"Evaluated with Stratified {cv_folds}-Fold Cross Validation · Sorted by {primary_metric}")

        # Styled dataframe
        st.dataframe(
            leaderboard.style.highlight_max(subset=["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"], color="#155e75")
                             .highlight_min(subset=["Runtime (s)"], color="#064e3b")
                             .format({"Accuracy": "{:.4f}", "ROC-AUC": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "Runtime (s)": "{:.3f}"}),
            use_container_width=True
        )

        champ_name = automl.champion_name
        best_score = leaderboard.iloc[0][primary_metric]

        st.markdown(f"""
        <div class="card-box" style="border-left: 5px solid #10b981;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="badge-champion">⭐ CHAMPION SELECTED</span>
                    <h3 style="margin: 6px 0 2px 0; color: white;">{champ_name}</h3>
                    <p style="margin: 0; color: #9ca3af; font-size: 13px;">
                        Achieved best <strong>{primary_metric}</strong> score of <strong>{best_score:.4f}</strong>. Preprocessing pipeline & hyperparameter refinement fitted.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visual Diagnostics
        st.markdown("### 📈 Model Diagnostics & Interpretability")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.markdown("##### 🔍 Champion Feature Importance")
            df_imp = automl.get_feature_importances()
            if not df_imp.empty:
                top_feats = df_imp.head(8).sort_values(by="Importance", ascending=True)
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                fig_imp.patch.set_facecolor("#111827")
                ax_imp.set_facecolor("#111827")
                ax_imp.barh(top_feats["Feature"], top_feats["Importance"], color="#38bdf8", edgecolor="#0284c7")
                ax_imp.set_xlabel("Importance Weight", color="#94a3b8")
                ax_imp.tick_params(colors="#e2e8f0")
                ax_imp.spines["top"].set_visible(False)
                ax_imp.spines["right"].set_visible(False)
                ax_imp.spines["left"].set_color("#374151")
                ax_imp.spines["bottom"].set_color("#374151")
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)
            else:
                st.info("Feature importance not directly exposed for this model type.")

        with vcol2:
            st.markdown("##### 🎯 Champion Confusion Matrix")
            y_pred = automl.predict(X)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            fig_cm.patch.set_facecolor("#111827")
            ax_cm.set_facecolor("#111827")
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                xticklabels=[meta["negative_label"][:10], meta["positive_label"][:10]],
                yticklabels=[meta["negative_label"][:10], meta["positive_label"][:10]]
            )
            ax_cm.set_title(f"Train/Fit Confusion Matrix", color="#f8fafc", fontsize=11, fontweight="bold")
            ax_cm.set_ylabel("Actual Label", color="#94a3b8")
            ax_cm.set_xlabel("Predicted Label", color="#94a3b8")
            ax_cm.tick_params(colors="#e2e8f0")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        # Live Inference Tester
        st.markdown("---")
        st.markdown("### 🧪 Live Inference / Record Tester")
        st.write("Modify feature values below to get real-time predictions from the champion model:")

        test_row = X.iloc[[0]].copy()
        test_cols = st.columns(min(4, len(X.columns)))
        for idx, col in enumerate(X.columns[:8]):
            curr_c = test_cols[idx % min(4, len(X.columns))]
            val = test_row.iloc[0][col]
            if isinstance(val, (int, np.integer)):
                new_v = curr_c.number_input(f"{col}", value=int(val), key=f"inp_{col}")
                test_row.at[test_row.index[0], col] = new_v
            elif isinstance(val, (float, np.floating)):
                new_v = curr_c.number_input(f"{col}", value=float(val), key=f"inp_{col}")
                test_row.at[test_row.index[0], col] = new_v
            else:
                options = list(X[col].dropna().unique())
                new_v = curr_c.selectbox(f"{col}", options=options, index=options.index(val) if val in options else 0, key=f"inp_{col}")
                test_row.at[test_row.index[0], col] = new_v

        live_pred = automl.predict(test_row)[0]
        live_proba = automl.predict_proba(test_row)[0]
        pos_prob = live_proba[1]

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            if live_pred == 1:
                st.error(f"Prediction: **{meta['positive_label']}**")
            else:
                st.success(f"Prediction: **{meta['negative_label']}**")
        with p_col2:
            st.progress(float(pos_prob))
            st.caption(f"Predicted Probability of {meta['positive_label']}: **{pos_prob*100:.1f}%**")

        # Model Export
        st.markdown("---")
        m_buf = io.BytesIO()
        joblib.dump(automl.champion_pipeline, m_buf)
        m_buf.seek(0)
        st.download_button(
            label="💾 Download Champion Model Pipeline (.pkl)",
            data=m_buf,
            file_name="champion_model.pkl",
            mime="application/octet-stream"
        )


with tab_code:
    st.subheader("💻 Pure Python Machine Learning Source Code")
    st.write("Browse production-ready ML scripts configured with Streamlit Cloud import protection and Scikit-Learn pipelines.")

    code_file = st.selectbox(
        "Select Python Source File",
        options=["pipeline.py", "train.py", "evaluate.py", "dataset.py", "app.py", "requirements.txt"]
    )
    code_content = load_code_file(code_file)
    st.code(code_content, language="python" if not code_file.endswith(".txt") else "text")

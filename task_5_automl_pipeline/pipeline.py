import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class AutoMLPipeline:
    """
    Automated Machine Learning (AutoML) Pipeline Engine.
    
    Provides end-to-end automation:
    - Automated feature type discovery (numerical vs categorical)
    - Automated imputation, standard scaling, and one-hot encoding
    - 5-Fold Stratified Cross-Validation benchmarking across 6+ model architectures
    - Multi-metric Leaderboard ranking (Accuracy, ROC-AUC, F1, Precision, Recall, Runtime)
    - Automated Champion Selection & Hyperparameter Refinement
    - Feature Importance extraction & model serialization
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.encoded_feature_names: List[str] = []
        
        self.leaderboard: Optional[pd.DataFrame] = None
        self.models_dict: Dict[str, Any] = {}
        self.fitted_pipelines: Dict[str, Pipeline] = {}
        
        self.champion_name: Optional[str] = None
        self.champion_pipeline: Optional[Pipeline] = None
        self.champion_params: Dict[str, Any] = {}
        self.benchmark_results: Dict[str, Dict[str, Any]] = {}

    def _inspect_and_build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Inspects tabular columns and builds an optimal Scikit-Learn ColumnTransformer.
        """
        self.numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(X.select_dtypes(exclude=[np.number]).columns)

        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Support sparse=False or sparse_output=False depending on scikit-learn version
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe)
        ])

        transformers = []
        if self.numeric_cols:
            transformers.append(("num", num_transformer, self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", cat_transformer, self.categorical_cols))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        return self.preprocessor

    def _get_model_candidates(self) -> Dict[str, Any]:
        """
        Returns a rich suite of 7 diverse classification models with stable hyperparameter defaults.
        """
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4, random_state=self.random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4, random_state=self.random_state
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=self.random_state
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=80, learning_rate=0.8, random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6, min_samples_split=6, random_state=self.random_state
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=7, weights="distance"
            )
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        primary_metric: str = "F1",
        cv_folds: int = 5,
        tune_champion: bool = True
    ) -> pd.DataFrame:
        """
        Runs the full AutoML loop:
        1. Fit preprocessor
        2. 5-fold CV across all candidates
        3. Build leaderboard
        4. Select & tune champion
        5. Fit final pipeline
        """
        self._inspect_and_build_preprocessor(X)
        models = self._get_model_candidates()
        self.models_dict = models
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]

        leaderboard_rows = []
        self.fitted_pipelines = {}
        self.benchmark_results = {}

        # Benchmarking loop
        for name, model in models.items():
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("classifier", model)
            ])

            t0 = time.time()
            try:
                cv_res = cross_validate(
                    pipe,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=False
                )
                fit_time = time.time() - t0

                acc = float(np.mean(cv_res["test_accuracy"]))
                auc = float(np.mean(cv_res["test_roc_auc"]))
                f1 = float(np.mean(cv_res["test_f1"]))
                prec = float(np.mean(cv_res["test_precision"]))
                rec = float(np.mean(cv_res["test_recall"]))
            except Exception as e:
                # Fallback for models that might struggle on specific subsets
                fit_time = time.time() - t0
                acc, auc, f1, prec, rec = 0.5, 0.5, 0.5, 0.5, 0.5

            # Fit on full dataset for immediate inference readiness
            pipe.fit(X, y)
            self.fitted_pipelines[name] = pipe

            row = {
                "Model": name,
                "Accuracy": np.round(acc, 4),
                "ROC-AUC": np.round(auc, 4),
                "F1": np.round(f1, 4),
                "Precision": np.round(prec, 4),
                "Recall": np.round(rec, 4),
                "Runtime (s)": np.round(fit_time, 3)
            }
            leaderboard_rows.append(row)
            self.benchmark_results[name] = row

        # Build Leaderboard DataFrame
        df_lb = pd.DataFrame(leaderboard_rows)
        sort_col = primary_metric if primary_metric in df_lb.columns else "F1"
        df_lb = df_lb.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        df_lb.index = df_lb.index + 1
        df_lb.index.name = "Rank"
        self.leaderboard = df_lb

        # Select Champion
        self.champion_name = df_lb.iloc[0]["Model"]
        champion_base_pipe = self.fitted_pipelines[self.champion_name]

        # Automated Hyperparameter Refinement
        if tune_champion:
            self._tune_champion(X, y, champion_base_pipe, self.champion_name, primary_metric)
        else:
            self.champion_pipeline = champion_base_pipe
            self.champion_params = champion_base_pipe.named_steps["classifier"].get_params()

        # Cache encoded feature names
        try:
            ohe_step = self.preprocessor.named_transformers_.get("cat", None)
            cat_feature_names = []
            if ohe_step and hasattr(ohe_step.named_steps["onehot"], "get_feature_names_out"):
                cat_feature_names = list(ohe_step.named_steps["onehot"].get_feature_names_out(self.categorical_cols))
            self.encoded_feature_names = self.numeric_cols + cat_feature_names
        except Exception:
            self.encoded_feature_names = [f"feat_{i}" for i in range(len(self.numeric_cols) + len(self.categorical_cols))]

        return self.leaderboard

    def _tune_champion(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_pipe: Pipeline,
        model_name: str,
        primary_metric: str
    ):
        """
        Runs targeted, rapid hyperparameter grid search on the champion model.
        """
        param_grids: Dict[str, Dict[str, List[Any]]] = {
            "Random Forest": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [8, 14, None],
                "classifier__min_samples_split": [2, 4]
            },
            "Gradient Boosting": {
                "classifier__n_estimators": [80, 120],
                "classifier__learning_rate": [0.05, 0.1, 0.15],
                "classifier__max_depth": [3, 5]
            },
            "Extra Trees": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [8, 14, None]
            },
            "Logistic Regression": {
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__solver": ["lbfgs", "saga"]
            },
            "AdaBoost": {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.5, 0.8, 1.0]
            },
            "Decision Tree": {
                "classifier__max_depth": [4, 6, 8],
                "classifier__min_samples_split": [4, 8]
            },
            "K-Nearest Neighbors": {
                "classifier__n_neighbors": [5, 7, 9],
                "classifier__weights": ["uniform", "distance"]
            }
        }

        grid = param_grids.get(model_name, {})
        scoring_metric = "f1" if primary_metric.lower() == "f1" else ("roc_auc" if "auc" in primary_metric.lower() else "accuracy")

        if grid:
            try:
                search = GridSearchCV(
                    base_pipe,
                    param_grid=grid,
                    cv=3,
                    scoring=scoring_metric,
                    n_jobs=-1
                )
                search.fit(X, y)
                self.champion_pipeline = search.best_estimator_
                self.champion_params = search.best_params_
                return
            except Exception:
                pass

        # Fallback to base fitted pipeline
        self.champion_pipeline = base_pipe
        self.champion_params = base_pipe.named_steps["classifier"].get_params()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.champion_pipeline is None:
            raise ValueError("AutoMLPipeline is not fitted yet. Call fit() first.")
        return self.champion_pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.champion_pipeline is None:
            raise ValueError("AutoMLPipeline is not fitted yet. Call fit() first.")
        if hasattr(self.champion_pipeline, "predict_proba"):
            return self.champion_pipeline.predict_proba(X)
        # Decision function fallback
        df = self.champion_pipeline.decision_function(X)
        probs = 1 / (1 + np.exp(-df))
        return np.vstack([1 - probs, probs]).T

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Extracts feature importances (or absolute coefficients) from the champion.
        """
        if self.champion_pipeline is None:
            return pd.DataFrame()

        clf = self.champion_pipeline.named_steps["classifier"]
        importances = None

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])

        if importances is None:
            return pd.DataFrame()

        names = self.encoded_feature_names
        if len(names) != len(importances):
            names = [f"feature_{i}" for i in range(len(importances))]

        df_imp = pd.DataFrame({
            "Feature": names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Normalize to percentage
        total = df_imp["Importance"].sum()
        if total > 0:
            df_imp["Relative_Weight_%"] = np.round((df_imp["Importance"] / total) * 100, 2)
        return df_imp

    def evaluate_champion_detailed(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics on unseen holdout test split.
        """
        if self.champion_pipeline is None:
            raise ValueError("Model is not fitted.")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        return {
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }
        }

    def save(self, filepath: str = "models/champion_model.pkl"):
        """
        Serializes the complete champion pipeline and metadata to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_payload = {
            "pipeline": self.champion_pipeline,
            "champion_name": self.champion_name,
            "champion_params": self.champion_params,
            "leaderboard": self.leaderboard,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "encoded_feature_names": self.encoded_feature_names,
            "random_state": self.random_state
        }
        joblib.dump(save_payload, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str = "models/champion_model.pkl") -> "AutoMLPipeline":
        """
        Deserializes a saved AutoMLPipeline from disk.
        """
        payload = joblib.load(filepath)
        instance = cls(random_state=payload.get("random_state", 42))
        instance.champion_pipeline = payload["pipeline"]
        instance.champion_name = payload["champion_name"]
        instance.champion_params = payload.get("champion_params", {})
        instance.leaderboard = payload.get("leaderboard", None)
        instance.numeric_cols = payload.get("numeric_cols", [])
        instance.categorical_cols = payload.get("categorical_cols", [])
        instance.encoded_feature_names = payload.get("encoded_feature_names", [])
        return instance

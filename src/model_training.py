import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

# Traditional learners
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

# Optional advanced learners
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# Project imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.config import TECHNICAL_SKILLS, PERSONALITY_TRAITS, CAREER_CATEGORIES, MODEL_PATH
from src.logging_config import get_logger

# Suppress convergence warnings from MLP
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = get_logger("enhanced_model_training")


class EnhancedModelTrainer:
    """
    Train, compare & autotune multiple ML algorithms for the career-prediction
    problem. The best model by CV accuracy is automatically persisted to disk.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        self.scaler = StandardScaler()
        self.label_enc = LabelEncoder()
        self.feature_names = (
            [s for s, _ in TECHNICAL_SKILLS] + [t for t, _ in PERSONALITY_TRAITS]
        )
        self.best_model = None
        self.results_ = {}  # name → (mean_acc, std_acc)

    @staticmethod
    def create_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Synthetic data generator (replace with real data loader)."""
        rng = np.random.default_rng(seed)
        tech = {
            skill: rng.integers(1, 8, size=n_samples)  # 1-7 scale
            for skill, _ in TECHNICAL_SKILLS
        }
        traits = {
            trait: rng.random(n_samples)  # 0-1
            for trait, _ in PERSONALITY_TRAITS
        }
        careers = rng.choice(CAREER_CATEGORIES, size=n_samples)
        df = pd.DataFrame({**tech, **traits, "career": careers})
        return df

    def _prepare_xy(self, df: pd.DataFrame):
        # Normalize technical skills to 0-1
        for skill, _ in TECHNICAL_SKILLS:
            df[skill] = df[skill] / 7.0
        X = df[self.feature_names].values
        y = self.label_enc.fit_transform(df["career"].values)
        return X, y

    def _model_zoo(self):
        """Return dict {name: (estimator, param_grid | None)}"""
        zoo = {
            "LogReg": (
                LogisticRegression(max_iter=1000, n_jobs=-1),
                {
                    "classifier__C": [0.1, 1, 10],
                    "classifier__penalty": ["l2"],
                    "classifier__solver": ["lbfgs", "saga"],
                },
            ),
            "KNN": (
                KNeighborsClassifier(),
                {
                    "classifier__n_neighbors": [3, 5, 11],
                    "classifier__weights": ["uniform", "distance"],
                },
            ),
            "SVM-RBF": (
                SVC(kernel="rbf", probability=True),
                {
                    "classifier__C": [0.1, 1, 10],
                    "classifier__gamma": ["scale", 0.01, 0.1],
                },
            ),
            "DecisionTree": (
                DecisionTreeClassifier(random_state=self.random_state),
                {
                    "classifier__max_depth": [None, 5, 10],
                    "classifier__min_samples_split": [2, 5, 10],
                },
            ),
            "NaiveBayes": (GaussianNB(), None),
            "QDA": (QuadraticDiscriminantAnalysis(), None),
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
                {
                    "classifier__max_depth": [None, 10, 25],
                    "classifier__min_samples_split": [2, 5],
                },
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(random_state=self.random_state),
                {
                    "classifier__n_estimators": [100, 200],
                    "classifier__learning_rate": [0.05, 0.1],
                },
            ),
            "AdaBoost": (
                AdaBoostClassifier(random_state=self.random_state),
                {
                    "classifier__n_estimators": [100, 200, 400],
                    "classifier__learning_rate": [0.1, 0.5, 1.0],
                },
            ),
            "MLP": (
                MLPClassifier(max_iter=1000, random_state=self.random_state),
                {
                    "classifier__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                    "classifier__alpha": [1e-4, 1e-3],
                    "classifier__max_iter": [500, 1000, 2000],
                },
            ),
        }
        if XGBClassifier is not None:
            zoo["XGBoost"] = (
                XGBClassifier(
                    eval_metric="mlogloss",
                    objective="multi:softprob",
                    num_class=len(CAREER_CATEGORIES),
                    random_state=self.random_state,
                    n_jobs=-1,
                    use_label_encoder=False,
                ),
                {
                    "classifier__learning_rate": [0.05, 0.1],
                    "classifier__max_depth": [3, 6, 9],
                    "classifier__n_estimators": [200, 500],
                    "classifier__subsample": [0.7, 1.0],
                },
            )
        if LGBMClassifier is not None:
            zoo["LightGBM"] = (
                LGBMClassifier(
                    objective="multiclass",
                    num_class=len(CAREER_CATEGORIES),
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
                {
                    "classifier__learning_rate": [0.05, 0.1],
                    "classifier__max_depth": [-1, 10, 20],
                    "classifier__n_estimators": [300, 600],
                },
            )
        if CatBoostClassifier is not None:
            zoo["CatBoost"] = (
                CatBoostClassifier(
                    verbose=0,
                    random_state=self.random_state,
                    loss_function="MultiClass",
                ),
                {
                    "classifier__depth": [4, 6, 8],
                    "classifier__learning_rate": [0.05, 0.1],
                    "classifier__iterations": [300, 600],
                },
            )
        return zoo

    def evaluate_all(self, X, y):
        """Cross-validate every model, store mean ± std accuracy."""
        zoo = self._model_zoo()
        logger.info(f"Evaluating {len(zoo)} models …")
        for name, (est, _) in zoo.items():
            pipe = Pipeline(steps=[("scaler", self.scaler), ("classifier", est)])
            cv_scores = cross_val_score(
                pipe, X, y, cv=self.cv, scoring="accuracy", n_jobs=-1
            )
            self.results_[name] = (cv_scores.mean(), cv_scores.std())
            logger.info(
                f"{name:12s} | acc = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )
        return self.results_

    def _tune_top(self, X, y, top_k: int = 3, n_iter: int = 30):
        """
        Hyper-parameter tune the top-k models by accuracy using RandomizedSearch
        (falls back to GridSearch if param_grid is small).
        """
        if not self.results_:
            raise RuntimeError("Run evaluate_all() first")
        ranked = sorted(
            self.results_.items(), key=lambda kv: kv[1][0], reverse=True
        )[:top_k]
        logger.info(f"Tuning top {top_k} models: {[r[0] for r in ranked]}")
        best_score = -np.inf
        best_pipe = None
        best_name = ""
        for name, _ in ranked:
            est, grid = self._model_zoo()[name]
            pipe = Pipeline([("scaler", self.scaler), ("classifier", est)])
            if not grid:
                logger.info(f"No hyperparameters for {name}, skipping tune")
                tuned_pipe = pipe
            else:
                total_params = sum(len(v) for v in grid.values())
                search_cv = (
                    RandomizedSearchCV(
                        pipe,
                        grid,
                        n_iter=min(n_iter, total_params),
                        scoring="accuracy",
                        cv=self.cv,
                        n_jobs=-1,
                        random_state=self.random_state,
                    )
                    if total_params > 20
                    else GridSearchCV(
                        pipe,
                        grid,
                        scoring="accuracy",
                        cv=self.cv,
                        n_jobs=-1,
                    )
                )
                search_cv.fit(X, y)
                tuned_pipe = search_cv.best_estimator_
                logger.info(f"{name} tuned acc = {search_cv.best_score_:.4f}")
            tuned_scores = cross_val_score(
                tuned_pipe, X, y, cv=self.cv, scoring="accuracy", n_jobs=-1
            )
            mean_acc = tuned_scores.mean()
            std_acc = tuned_scores.std()
            logger.info(f"{name:12s} | tuned acc = {mean_acc:.4f} ± {std_acc:.4f}")
            if mean_acc > best_score:
                best_score, best_pipe, best_name = mean_acc, tuned_pipe, name
        logger.info(f"Selected best model ➜ {best_name} ({best_score:.4f} CV accuracy)")
        self.best_model = best_pipe
        return best_name, best_score

    def final_fit(self, X, y):
        """Fit best model on the entire dataset."""
        if self.best_model is None:
            raise RuntimeError("Run _tune_top() first")
        self.best_model.fit(X, y)

    def save(self, path: str = MODEL_PATH):
        """Persist best model + label encoder + target classes to joblib file."""
        if self.best_model is None:
            raise RuntimeError("Model not trained yet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.best_model,
                "feature_names": self.feature_names,
                "target_classes": self.label_enc.classes_,  # actual class names
                "label_encoder": self.label_enc,           # for inverse transform
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def train_pipeline(self, df: pd.DataFrame = None):
        """Full zero-to-hero training pipeline."""
        if df is None:
            df = self.create_sample_data()
        X, y = self._prepare_xy(df)
        self.evaluate_all(X, y)
        self._tune_top(X, y, top_k=3, n_iter=25)
        self.final_fit(X, y)
        self.save()
        # Final hold-out evaluation (20 % split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        y_pred = self.best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Hold-out accuracy = {acc:.4f}")
        class_report = classification_report(y_test, y_pred, zero_division=0)
        logger.info("\n" + class_report)
        logger.info("\nConfusion matrix:\n%s", confusion_matrix(y_test, y_pred))
        return {
            "holdout_accuracy": acc,
            "classification_report": class_report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "cv_results": self.results_,
        }


if __name__ == "__main__":
    trainer = EnhancedModelTrainer(cv_folds=5, random_state=42)
    metrics = trainer.train_pipeline()
    print("\n==== FINAL RESULTS ====")
    print(f"Hold-out Accuracy: {metrics['holdout_accuracy']:.3f}")
    print("\nClassification Report:\n", metrics["classification_report"])

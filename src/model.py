from typing import Tuple, Dict
import os

import numpy as np
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

from src.config import SEED


def build_random_forest(n_estimators: int = 200) -> RandomForestClassifier:
    """
    Build a Random Forest classifier with fixed hyperparameters.

    The hyperparameters are intentionally kept constant to reduce
    confounding effects and ensure reproducibility across experiments.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=-1,
    )


def train_model(
    model: RandomForestClassifier,
    X_train,
    y_train,
) -> RandomForestClassifier:
    """
    Train the given model on the training data.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test,
    y_test,
) -> Dict[str, float]:
    """
    Evaluate the trained model on the test data.

    Returns a dictionary containing standard performance metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics


def get_feature_importances(
    model: RandomForestClassifier,
) -> np.ndarray:
    """
    Return feature importances from a trained Random Forest model.

    Note: For one-hot encoded features, importances correspond to
    expanded feature dimensions rather than original raw features.
    """
    return model.feature_importances_


def save_model(model: RandomForestClassifier, path: str) -> None:
    """
    Save a trained model to disk using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> RandomForestClassifier:
    """
    Load a trained model from disk.
    """
    return joblib.load(path)


def train_and_evaluate(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_test: spmatrix,
    y_test: np.ndarray,
    n_estimators: int = 200
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Convenience function that builds, trains, and evaluates
    a Random Forest classifier.

    Returns the trained model and evaluation metrics.
    """
    model = build_random_forest(n_estimators)
    model = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics

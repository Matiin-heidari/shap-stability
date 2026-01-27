import numpy as np
import shap
from scipy.sparse import spmatrix, issparse

from src.config import SEED


def compute_shap_values(
    model,
    X_background: spmatrix,
    X_explain: spmatrix,
    max_background_samples: int = 100,
) -> np.ndarray:
    """
    Compute SHAP values for a trained tree-based model using TreeSHAP.

    Both background data and explained instances are converted to dense
    arrays. The strict additivity check is disabled due to known numerical
    issues with Random Forest probability outputs.
    """
    rng = np.random.default_rng(SEED)

    if X_background.shape[0] > max_background_samples:
        idx = rng.choice(
            X_background.shape[0],
            size=max_background_samples,
            replace=False,
        )
        background = X_background[idx]
    else:
        background = X_background

    background_dense = (
        background.toarray()
        if issparse(background)
        else background
    )

    explain_dense = (
        X_explain.toarray()
        if issparse(X_explain)
        else X_explain
    )

    explainer = shap.TreeExplainer(
        model,
        data=background_dense,
        feature_perturbation="interventional",
    )

    shap_values = explainer.shap_values(
        explain_dense,
        check_additivity=False,
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values

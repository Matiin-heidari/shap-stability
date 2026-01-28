from typing import Dict
import numpy as np
from scipy.stats import spearmanr
from numpy.linalg import norm


def _select_positive_class(shap_values: np.ndarray) -> np.ndarray:
    """
    Select SHAP values for the positive class in binary classification.
    """
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def spearman_stability(
    shap_ref: np.ndarray,
    shap_perturbed: np.ndarray,
) -> float:
    """
    Compute average Spearman rank correlation between reference and
    perturbed SHAP values across instances.
    """
    shap_ref = _select_positive_class(shap_ref)
    shap_perturbed = _select_positive_class(shap_perturbed)

    correlations = []
    for i in range(shap_ref.shape[0]):
        corr, _ = spearmanr(shap_ref[i], shap_perturbed[i])
        correlations.append(corr)

    return float(np.nanmean(correlations))


def cosine_stability(
    shap_ref: np.ndarray,
    shap_perturbed: np.ndarray,
) -> float:
    """
    Compute average cosine similarity between reference and
    perturbed SHAP values across instances.
    """
    shap_ref = _select_positive_class(shap_ref)
    shap_perturbed = _select_positive_class(shap_perturbed)

    similarities = []
    for i in range(shap_ref.shape[0]):
        num = np.dot(shap_ref[i], shap_perturbed[i])
        denom = norm(shap_ref[i]) * norm(shap_perturbed[i])
        similarities.append(num / denom if denom > 0 else np.nan)

    return float(np.nanmean(similarities))


def attribution_variance(
    shap_values_list: np.ndarray,
) -> float:
    """
    Compute mean variance of SHAP values across perturbations.

    Expects an array of shape:
    (n_perturbations, n_samples, n_features, n_classes)
    """
    shap_values_list = _select_positive_class(shap_values_list)

    variances = np.var(shap_values_list, axis=0)
    return float(np.mean(variances))


def compute_stability_metrics(
    shap_ref: np.ndarray,
    shap_perturbed: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a set of stability metrics between reference and
    perturbed SHAP explanations.
    """
    return {
        "spearman": spearman_stability(shap_ref, shap_perturbed),
        "cosine": cosine_stability(shap_ref, shap_perturbed),
    }

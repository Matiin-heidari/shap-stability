from typing import Tuple
import numpy as np
from scipy.sparse import spmatrix, issparse

from src.config import SEED


def add_gaussian_noise(
    X,
    noise_std: float,
):
    """
    Add Gaussian noise to feature matrix.

    Supports both sparse and dense inputs.
    For sparse inputs, noise is added only to non-zero entries.
    """
    rng = np.random.default_rng(SEED)

    if issparse(X):
        X_noisy = X.copy().tolil()
        rows, cols = X_noisy.nonzero()
        noise = rng.normal(loc=0.0, scale=noise_std, size=len(rows))
        X_noisy[rows, cols] += noise
        return X_noisy.tocsr()

    # Dense case
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    return X + noise


def bootstrap_resample(
    X_train: spmatrix,
    y_train,
) -> Tuple[spmatrix, np.ndarray]:
    """
    Bootstrap resampling of the training data.
    """
    rng = np.random.default_rng(SEED)
    n_samples = X_train.shape[0]

    indices = rng.choice(n_samples, size=n_samples, replace=True)

    return X_train[indices], y_train.iloc[indices].to_numpy()
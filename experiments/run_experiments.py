import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.config import SEED
from src.data import load_adult_income, load_german_credit
from src.model import train_and_evaluate
from src.explain import compute_shap_values
from src.perturb import add_gaussian_noise, bootstrap_resample
from src.metrics import compute_stability_metrics, attribution_variance
from src.utils import set_global_seed, setup_logging


# =========================
# Experiment configuration
# =========================
N_EXPLAIN = 100
N_BOOTSTRAPS = 20
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.2]

# Ablation: number of trees
N_ESTIMATORS_LIST = [200, 50]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Experiment functions
# =========================
def run_noise_experiments(dataset: str) -> List[Dict]:
    """
    Evaluate SHAP stability under additive Gaussian noise
    with an ablation over the number of trees.
    """
    logging.info(f"Running noise experiments for dataset={dataset}")

    if dataset == "adult":
        X_train, X_test, y_train, y_test, _ = load_adult_income()
    elif dataset == "german":
        X_train, X_test, y_train, y_test, _ = load_german_credit()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    results = []

    for n_estimators in N_ESTIMATORS_LIST:
        logging.info(f"Training model with n_estimators={n_estimators}")

        model, _ = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            n_estimators=n_estimators,
        )

        shap_ref = compute_shap_values(
            model,
            X_background=X_train,
            X_explain=X_test[:N_EXPLAIN],
        )

        for noise in NOISE_LEVELS:
            logging.info(
                f"Noise level={noise} | n_estimators={n_estimators}"
            )

            if noise == 0.0:
                X_noisy = X_test[:N_EXPLAIN]
            else:
                X_noisy = add_gaussian_noise(
                    X_test[:N_EXPLAIN],
                    noise,
                )

            shap_noisy = compute_shap_values(
                model,
                X_background=X_train,
                X_explain=X_noisy,
            )

            metrics = compute_stability_metrics(
                shap_ref,
                shap_noisy,
            )

            results.append(
                {
                    "dataset": dataset,
                    "noise_std": noise,
                    "n_estimators": n_estimators,
                    **metrics,
                }
            )

    return results


def run_bootstrap_experiments(dataset: str) -> List[Dict]:
    """
    Evaluate SHAP stability under bootstrap resampling.
    """
    logging.info(f"Running bootstrap experiments for dataset={dataset}")

    if dataset == "adult":
        X_train, X_test, y_train, y_test, _ = load_adult_income()
    elif dataset == "german":
        X_train, X_test, y_train, y_test, _ = load_german_credit()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    shap_bootstrap = []

    for i in tqdm(range(N_BOOTSTRAPS), desc="Bootstraps"):
        X_bs, y_bs = bootstrap_resample(X_train, y_train)

        model_bs, _ = train_and_evaluate(
            X_bs,
            y_bs,
            X_test,
            y_test,
        )

        shap_bs = compute_shap_values(
            model_bs,
            X_background=X_bs,
            X_explain=X_test[:N_EXPLAIN],
        )

        shap_bootstrap.append(shap_bs)

    variance = attribution_variance(np.stack(shap_bootstrap))

    return [
        {
            "dataset": dataset,
            "bootstrap_variance": variance,
        }
    ]


# =========================
# Main entry point
# =========================
def main() -> None:
    setup_logging()
    set_global_seed(SEED)

    logging.info("Starting SHAP stability experiments")

    all_results: List[Dict] = []

    for dataset in ["adult", "german"]:
        all_results.extend(run_noise_experiments(dataset))
        all_results.extend(run_bootstrap_experiments(dataset))

    output_path = RESULTS_DIR / "stability_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"Results saved to {output_path}")
    logging.info("All experiments completed successfully")


if __name__ == "__main__":
    main()

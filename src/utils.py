import random
import numpy as np
import os

def set_global_seed(seed: int):
    """
    Set random seeds for all relevant libraries to ensure reproducibility.

    This function fixes sources of randomness in Python's random module,
    NumPy, and Python's hash-based operations. It should be called once
    at the beginning of each experiment run.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

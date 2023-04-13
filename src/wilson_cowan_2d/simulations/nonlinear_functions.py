import numpy as np
from numpy import ndarray


def decreasing_exponential(x: ndarray, σ: float) -> ndarray:
    """Computes the kernal for an number, assumes it's been normalized"""
    return 1/(2*σ) * np.exp(-x/σ)


def sigmoid(x: ndarray, β: float) -> ndarray:
    return 1/(1 + np.exp(-β * x))


def ssn_power_law(x: ndarray, n: float, k: float) -> ndarray:
    m = np.ma.masked_array(x, x <= 0, 0)
    return k * np.power(m.filled(), n)

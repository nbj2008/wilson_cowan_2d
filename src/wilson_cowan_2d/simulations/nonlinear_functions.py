from numpy import ndarray, exp as nexp


def decreasing_exponential(x: ndarray, σ: float) -> ndarray:
    """Computes the kernal for an number, assumes it's been normalized"""
    return 1/(2*σ) * nexp(-x/σ)


def sigmoid(x: ndarray, β: float) -> ndarray:
    return 1/(1 + nexp(-β * x))

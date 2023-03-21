from numpy import exp, mgrid, power, abs as nabs, sum
from .grids import Grid
from typing import Callable


class KernelGrid:
    def __init__(self, K: Callable, grid: Grid):
        self.K = K
        self._grid = grid

    @property
    def grid(self):
        return self.K(self._grid.grid)

    def __getitem__(self, key):
        return self.K(self._grid.__getitem__(key))


def decreasing_exponential(x: float, σ):
    """Computes the kernal for an number, assumes it's been normalized"""
    return 1/(2*σ) * exp(-x/σ)


def dec_exp_1d_K(size, σ):
    acs, dwn = mgrid[:size, :size]
    return decreasing_exponential(nabs((acs-dwn).reshape(size, size)), σ)


def get_normalized_1d_K(size, σ):
    kern = dec_exp_1d_K(size, σ)
    return (kern.T*power(sum(kern, axis=0), -1)).T

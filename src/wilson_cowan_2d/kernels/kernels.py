from numpy import exp, mgrid, power, abs as nabs, sum, ndarray, stack
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


def make_K_2_populations(K: Callable, σ: ndarray) -> Callable:
    def Kernel(grid: ndarray) -> ndarray:
        return stack([K(grid, σ[0]), K(grid, σ[1])])
    return Kernel


def dec_exp_1d_K(size: int, σ: float):
    acs, dwn = mgrid[:size, :size]
    return decreasing_exponential(nabs((acs-dwn).reshape(size, size)), σ)


def get_normalized_1d_K(size: int, σ: float):
    kern = dec_exp_1d_K(size, σ)
    return (kern.T*power(sum(kern, axis=0), -1)).T

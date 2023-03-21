from numpy import exp, ndarray, stack
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

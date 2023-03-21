from ..kernels.grids import Dist1DGrid, Dist2DGrid, UnifGrid

# Typing
from numpy import array, exp as nexp, stack
from .abstract_wc_kernel import WCDecExp,  WCKernelParam, WCKernel


class DefaultParams(WCKernelParam):
    def __init__(self, τ, size):
        super().__init__(
             A=array([[1, 1.5], [1, 0.25]]),
             Θ=array([0.125, 0.4]),
             F=lambda x: 1/(1 + nexp(-50 * x)),  # Sigmoid with β=50
             τ=τ, size=size)


class WCUnif(WCKernel):
    def _make_grid(self):
        self._grid = UnifGrid(self.size).grid

    @property
    def kernel_grid(self):
        return stack([self.grid, self.grid])


class WCDecExp1D(WCDecExp):
    def _make_grid(self):
        self._grid = Dist1DGrid(self.size).grid


class WCDecExp2D(WCDecExp):
    def _make_grid(self):
        self._grid = Dist2DGrid(self.size).grid

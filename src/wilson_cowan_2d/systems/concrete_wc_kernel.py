import numpy as np
from ..kernels.kernels import decreasing_exponential
from ..kernels.grids import Dist1DGrid, UnifGrid # , Dist2DGrid

# Typing
from numpy import array, stack, concatenate, ndarray, split, exp as nexp
from typing import Tuple, Callable
from .abstract_wc_kernel import WCDecExp,  WCKernelParam, WCKernel
from .concrete_solved_system import (UnifSolvedSystem, Local1DSolvedSystem,
                                     NonLocal1DSolvedSystem)


class DefaultParams(WCKernelParam):
    def __init__(self, τ: float, η: float, size: int) -> WCKernelParam:
        super().__init__(
             A=array([[1, 1.5], [1, 0.25]]),
             Θ=array([0.125, 0.4]),
             β=50,
             τ=τ, size=size,
             η=η)

    def F(self, other):
        return 1/(1 + nexp(-self.β * other))


class WCUnif(WCKernel):
    @property
    def _get_solved_system(self) -> Callable:
        return UnifSolvedSystem

    def _make_grid(self) -> ndarray:
        self._grid = UnifGrid(self.size).grid

    @property
    def kernel_grid(self) -> ndarray:
        return stack([self.grid, self.grid])

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        inp = inp.reshape(2, self.size)
        A = self.A * array([[1, -1], [1, -1]])  # To subtract ends
        F = self.F
        Θ = self.θ
        τ = self.τ
        KK = self.kernel_grid

        x = ((F((A @ (KK @ inp.T)[0].T) - Θ) - inp) * τ)
        return x.ravel()


class WCDecExpLocal1D(WCDecExp):
    @property
    def _get_solved_system(self) -> Callable:
        return Local1DSolvedSystem

    def _make_grid(self) -> ndarray:
        self._grid = Dist1DGrid(self.size).grid

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        u, v, w, z = split(inp.reshape(4*self.size, 1), 4)
        σe, σi = self.σ
        F = self.F
        A = self.A
        θ = self.θ
        τe, τi = self.τ
        η = self.η
        abss = np.abs(np.linspace(-self.size, self.size, 2*self.size-1))
        DEe = decreasing_exponential(abss, σe)
        # DEe /= DEe.sum()
        DEi = decreasing_exponential(abss, σi)
        # DEi /= DEi.sum()
        Ke = np.convolve(DEe , u.ravel(),mode='valid').reshape(self.size, 1)  # self.kernel_grid[0]
        Ki = np.convolve(DEi, v.ravel(), mode='valid').reshape(self.size, 1)  # self.kernel_grid[1]
        # print(Ke.shape, Ki.shape, v.shape)

        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * Ki - θ[0])))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] *  Ke - A[1, 1] * Ki - θ[1]))\
            .reshape(v.shape)

        dw = z

        dz = (w-u)/σe**2

        return concatenate((du, dv, dw, dz)).ravel()


class WCDecExpNonLocal1D(WCDecExp):
    @property
    def _get_solved_system(self) -> Callable:
        return NonLocal1DSolvedSystem

    def _make_grid(self) -> ndarray:
        self._grid = Dist1DGrid(self.size).grid

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        u, v, w, z, q, p = split(inp.reshape(6*self.size, 1), 6)
        σe, σi = self.σ
        F = self.F
        A = self.A
        θ = self.θ
        τ = self.τ
        η = self.η

        du = 1/η*(-u + F((A[0, 0] * w - A[0, 1] * v - θ[0])))\
            .reshape(u.shape)

        dv = 1/(η*τ)*(-v + F(A[1, 0] * w - A[1, 1] * v - θ[1]))\
            .reshape(v.shape)

        dw = z

        dz = (w-u)/σe**2

        dq = p

        dp = (q-v)/σi**2

        return concatenate((du, dv, dw, dz, dq, dp)).ravel()


# class WCDecExp2D(WCDecExp):
    # def _make_grid(self) -> ndarray:
        # self._grid = Dist2DGrid(self.size).grid

    # def update(self, t, w):
        # w = w.reshape(2, self.size)
        # A = self.A * array([[1, -1], [1, -1]])  # To subtract ends
        # F = self.F
        # Θ = self.θ
        # τ = self.τ
        # KK = self.kernel_grid

        # x = ((F((A @ (KK @ w.T)[0].T) - Θ) - w) * τ)
        # return x.ravel()

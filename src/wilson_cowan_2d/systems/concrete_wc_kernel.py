from ..kernels.grids import Dist1DGrid, Dist2DGrid, UnifGrid

# Typing
from numpy import array, stack, concatenate, ndarray, split, exp as nexp
from typing import Tuple, Callable
from .abstract_wc_kernel import (WCDecExp,  WCKernelParam,
                                 WCKernel, SolvedSystem)


class DefaultParams(WCKernelParam):
    def __init__(self, τ: float, η: float, size: int) -> WCKernelParam:
        super().__init__(
             A=array([[1, 1.5], [1, 0.25]]),
             Θ=array([0.125, 0.4]),
             F=lambda x: 1/(1 + nexp(-50 * x)),  # Sigmoid with β=50
             τ=τ, size=size,
             η=η)


class UnifSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return array([inp[0] for inp in self._inps])

    @property
    def v(self) -> ndarray:
        return array([inp[1] for inp in self._inps])

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])


class Local1DSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return array([inp[0] for inp in self._inps])

    @property
    def v(self) -> ndarray:
        return array([inp[1] for inp in self._inps])

    @property
    def w(self) -> ndarray:
        return array([inp[2] for inp in self._inps])

    @property
    def z(self) -> ndarray:
        return array([inp[3] for inp in self._inps])

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])

    @property
    def dw(self) -> ndarray:
        return array([dinp[2] for dinp in self._dinps])

    @property
    def dz(self) -> ndarray:
        return array([dinp[3] for dinp in self._dinps])


class NonLocal1DSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return array([inp[0] for inp in self._inps])

    @property
    def v(self) -> ndarray:
        return array([inp[1] for inp in self._inps])

    @property
    def w(self) -> ndarray:
        return array([inp[2] for inp in self._inps])

    @property
    def z(self) -> ndarray:
        return array([inp[3] for inp in self._inps])

    @property
    def q(self) -> ndarray:
        return array([inp[4] for inp in self._inps])

    @property
    def p(self) -> ndarray:
        return array([inp[5] for inp in self._inps])

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])

    @property
    def dw(self) -> ndarray:
        return array([dinp[2] for dinp in self._dinps])

    @property
    def dz(self) -> ndarray:
        return array([dinp[3] for dinp in self._dinps])

    @property
    def dq(self) -> ndarray:
        return array([dinp[4] for dinp in self._dinps])

    @property
    def dp(self) -> ndarray:
        return array([dinp[5] for dinp in self._dinps])


class WCUnif(WCKernel):
    def _make_solved_sytem(self) -> Callable:
        return UnifSolvedSystem

    def _make_grid(self) -> ndarray:
        self._grid = UnifGrid(self.size).grid

    @property
    def kernel_grid(self) -> ndarray:
        return stack([self.grid, self.grid])

    @property
    def inital_inp(self) -> ndarray:
        return concatenate([self.u, self.v])

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
    def _make_solved_system(self) -> Callable:
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
        τ = self.τ
        η = self.η

        du = 1/η*(-u + F((A[0, 0] * w - A[0, 1] * v - θ[0])))\
            .reshape(u.shape)

        dv = 1/(η*τ)*(-v + F(A[1, 0] * w - A[1, 1] * v - θ[1]))\
            .reshape(v.shape)

        dw = z

        dz = (w-u)/σe**2

        return concatenate((du, dv, dw, dz)).ravel()


class WCDecExpNonLocal1D(WCDecExp):
    def _make_solved_system(self) -> Callable:
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


class WCDecExp2D(WCDecExp):
    def _make_grid(self) -> ndarray:
        self._grid = Dist2DGrid(self.size).grid

    # def update(self, t, w):
        # w = w.reshape(2, self.size)
        # A = self.A * array([[1, -1], [1, -1]])  # To subtract ends
        # F = self.F
        # Θ = self.θ
        # τ = self.τ
        # KK = self.kernel_grid

        # x = ((F((A @ (KK @ w.T)[0].T) - Θ) - w) * τ)
        # return x.ravel()

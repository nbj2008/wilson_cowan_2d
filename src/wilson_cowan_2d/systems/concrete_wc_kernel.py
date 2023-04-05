import numpy as np
from ..kernels.kernels import decreasing_exponential
from .stability import calc_AA

# Typing
from numpy import array, stack, concatenate, ndarray, split, exp as nexp
from typing import Tuple, Callable
from .abstract_wc_kernel import WCDecExp,  WCKernelParam, WCKernel, Param
from .concrete_solved_system import (UnifSolvedSystem, Local1DSolvedSystem,
                                     NonLocal1DSolvedSystem)


class DefaultParams(WCKernelParam):
    """Default set of Parameters used in Harris 2018"""
    def __init__(self, τ: float, η: float, size: int) -> WCKernelParam:
        super().__init__(
             A=array([[1, 1.5], [1, 0.25]]),
             Θ=array([0.125, 0.4]),
             β=50,
             τ=τ, size=size,
             η=η)

    def F(self, other):
        return 1/(1 + nexp(-self.β * other))


class MondronomyParams(DefaultParams):
    """Extends default parameters to record ω"""
    def __init__(self, τ: float, η: float, size: int, ω: float) -> WCKernelParam:
        super().__init__(τ=τ, size=size, η=η)
        self.ω = ω

    def F(self, other):
        return 1/(1 + nexp(-self.β * other))


class WCUnif(WCKernel):
    """Neural system using uniform kernel (Space-Clamped)"""
    @property
    def _get_solved_system(self) -> Callable:
        return UnifSolvedSystem

    def _make_grid(self) -> ndarray:
        self._grid = UnifGrid(self.size).grid

    @property
    def kernel_grid(self) -> ndarray:
        return stack([self.grid, self.grid])

    @property
    def grid(self):
        return self._grid

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        self._make_grid()
        inp = inp.reshape(2, self.size)
        A = self.A * array([[1, -1], [1, -1]])  # To subtract ends
        F = self.F
        Θ = self.θ
        τ = self.τ
        KK = self.kernel_grid

        x = ((F((A @ (KK @ inp.T)[0].T) - Θ) - inp) * τ)
        return x.ravel()


class WCDecExpStatic1D(WCDecExp):
    """Neural system to find Turing Pattern"""
    @property
    def _get_solved_system(self) -> Callable:
        return Local1DSolvedSystem

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        u, v = split(inp.reshape(2*self.size, 1), 2)
        σe, σi = self.σ
        F = self.F
        A = self.A
        θ = self.θ
        τe, τi = self.τ
        η = self.η

        x_lm = self.size/2
        mr_lm = self.size - 1
        abss = np.abs(np.linspace(-x_lm, x_lm, 2*mr_lm + 1))
        dx = 2*x_lm/self.size

        DEe = decreasing_exponential(abss, σe)
        DEi = decreasing_exponential(abss, σi)

        Ke = dx*np.convolve(DEe, u.ravel(), mode='valid').reshape(self.size, 1)
        Ki = dx*np.convolve(DEi, v.ravel(), mode='valid').reshape(self.size, 1)

        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * Ki - θ[0])))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * Ki - θ[1]))\
            .reshape(v.shape)

        return concatenate((du, dv)).ravel()


class WCDecExpTravelLocal1D(WCDecExp):
    """Simulating traveling wave in localized inhibition neural system"""
    @property
    def _get_solved_system(self) -> Callable:
        return NonLocal1DSolvedSystem

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        u, v = split(inp.reshape(2*self.size, 1), 2)
        σe, σi = self.σ
        F = self.F
        A = self.A
        θ = self.θ
        τe, τi = self.τ
        η = self.η

        abss = np.abs(np.linspace(-self.size, self.size, 2*self.size-1))
        DEe = decreasing_exponential(abss, σe)

        Ke = np.convolve(DEe, u.ravel(), mode='valid').reshape(self.size, 1)

        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * v - θ[0])))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * v - θ[1]))\
            .reshape(v.shape)

        return concatenate((du, dv)).ravel()


class WCDecExpTravelNonLocal1D(WCDecExp):
    """Simulating traveling wave in non-localized inhibition neural system"""
    @property
    def _get_solved_system(self) -> Callable:
        return NonLocal1DSolvedSystem

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        # Model Params
        u, v = split(inp.reshape(2*self.size, 1), 2)
        σe, σi = self.σ
        F = self.F
        A = self.A
        θ = self.θ
        τe, τi = self.τ
        η = self.η

        # Space Params
        x_lm = 21  # Found heuristically. No rational for limit from literature
        mr_lm = self.size-1
        dx = 2*x_lm/self.size
        abss = np.abs(np.linspace(-x_lm, x_lm, self.size))

        DEe = decreasing_exponential(abss, σe)
        u_mirror = np.concatenate([np.flip(u).ravel()[:mr_lm],
                                   u.ravel(), np.flip(u).ravel()[1:]])
        DEi = decreasing_exponential(abss, σi)
        v_mirror = np.concatenate([np.flip(v).ravel()[:mr_lm],
                                   v.ravel(), np.flip(v).ravel()[1:]])

        Ke = dx*np.convolve(DEe, u_mirror, mode='same')[mr_lm:2*mr_lm+1]\
            .reshape(self.size, 1)
        Ki = dx*np.convolve(DEi, v_mirror, mode='same')[mr_lm:2*mr_lm+1]\
            .reshape(self.size, 1)

        # ODE equations
        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * Ki - θ[0])))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * Ki - θ[1]))\
            .reshape(v.shape)

        return concatenate((du, dv)).ravel()


class OLDWCDecExpTravelNonLocal1D(WCDecExp):
    """Just putting this here so it's on the git log somewhere. Has commented
    out code from previous implementation
    """
    @property
    def _get_solved_system(self) -> Callable:
        return NonLocal1DSolvedSystem

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        u, v = split(inp.reshape(2*self.size, 1), 2)
        σe, σi = self.σ
        F = self.F
        A = self.A
        θ = self.θ
        τe, τi = self.τ
        η = self.η

        # abss = np.abs(np.linspace(-self.size/2, self.size/2, self.size))
        x_lm = 21
        mr_lm = self.size-1
        dx = 2*x_lm/self.size
        abss = np.abs(np.linspace(-x_lm, x_lm, self.size))

        DEe = decreasing_exponential(abss, σe)
        u_mirror = np.concatenate([np.flip(u).ravel()[:mr_lm],
                                   u.ravel(), np.flip(u).ravel()[1:]])
        DEi = decreasing_exponential(abss, σi)
        v_mirror = np.concatenate([np.flip(v).ravel()[:mr_lm],
                                   v.ravel(), np.flip(v).ravel()[1:]])

        Ke = dx*np.convolve(DEe, u_mirror, mode='same')[mr_lm:2*mr_lm+1]\
            .reshape(self.size, 1)
        Ki = dx*np.convolve(DEi, v_mirror, mode='same')[mr_lm:2*mr_lm+1]\
            .reshape(self.size, 1)
        # Ke = np.convolve(DEe, u.ravel(), mode='valid').reshape(self.size, 1)
        # Ki = np.convolve(DEi, v.ravel(), mode='valid').reshape(self.size, 1)

        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * Ki - θ[0])))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * Ki - θ[1]))\
            .reshape(v.shape)

        return concatenate((du, dv)).ravel()


# TODO: Actually get this to work
class WCDecExpMondronomy(WCDecExp):
    """System for determining the Mondronomy Matrix"""
    def __init__(self, inp: Tuple[ndarray], param: Param, σ: ndarray):
        super().__init__(inp, param, σ)
        self._simple = True

    @property
    def _get_solved_system(self):
        pass

    @property
    def initial_inp_ravel(self):
        return self._init_inp

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        # X = inp.reshape(2, 2)  # inp[0:4].reshape(2, 2)
        X = inp[0:4].reshape(2, 2)
        u, v = inp[4:6]
        σe, σi = self.σ
        F = self.F
        A = self.A
        θ = self.θ
        τe, τi = self.τ
        η = self.η
        ω = self.param.ω

        # # Iterate U an V values
        du = 1/(η*τe)*(-u + F((A[0, 0] * u - A[0, 1] * v - θ[0])))

        dv = 1/(η*τi)*(-v + F(A[1, 0] * u - A[1, 1] * v - θ[1]))

        # Calculate X values
        dX = np.zeros_like(X)
        if t >= 5:
            AA = calc_AA(self.param, u, v, σe, σi, τe, τi, ω)
            dX = AA @ X

        # return dX.ravel()
        return concatenate((dX.ravel(), np.array(du), np.array(dv))).ravel()

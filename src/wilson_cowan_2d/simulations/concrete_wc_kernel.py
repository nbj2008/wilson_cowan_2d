import numpy as np
from numpy import concatenate, ndarray, split
from scipy import linalg as la
from scipy.signal import convolve2d
from .nonlinear_functions import decreasing_exponential
from ..analysis.stability import calc_AA

# Typing
from typing import Tuple
from .wc_kernel import WCKernel
from . import Param


class WCUnif(WCKernel):
    """Neural system using uniform kernel (Space-Clamped)"""
    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        u, v = inp
        F = self.F
        A, (θe, θi), (τe, τi), η, _ = self.param.derivative_tuple

        du = 1/(η*τe)*(-u + F((A[0, 0] * u - A[0, 1] * v - θe)))\
            .reshape(u.shape)
        dv = 1/(η*τi)*(-v + F(A[1, 0] * u - A[1, 1] * v - θi))\
            .reshape(v.shape)

        return np.concatenate((du, dv))


class WCDecExpStatic1D(WCKernel):
    """Neural system to find Turing Pattern"""
    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        u, v = split(inp.reshape(2*self.size, 1), 2)
        F = self.F
        A, (θe, θi), (τe, τi), η, (σe, σi) = self.param.derivative_tuple

        x_lm = self.size/2
        mr_lm = self.size - 1
        abss = np.abs(np.linspace(-x_lm, x_lm, 2*mr_lm + 1))
        dx = 2*x_lm/self.size

        DEe = decreasing_exponential(abss, σe)
        DEi = decreasing_exponential(abss, σi)

        Ke = dx*np.convolve(DEe, u.ravel(), mode='valid').reshape(self.size, 1)
        Ki = dx*np.convolve(DEi, v.ravel(), mode='valid').reshape(self.size, 1)

        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * Ki - θe)))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * Ki - θi))\
            .reshape(v.shape)

        return concatenate((du, dv)).ravel()


class WCDecExpTravelLocal1D(WCKernel):
    """Simulating traveling wave in localized inhibition neural system"""
    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        u, v = split(inp.reshape(2*self.size, 1), 2)
        F = self.F
        A, (θe, θi), (τe, τi), η, (σe, σi) = self.param.derivative_tuple

        # Space Param
        x_lm = 21
        mr_lm = self.size-1
        dx = 2*x_lm/self.size

        abss = np.abs(np.linspace(-x_lm, x_lm, self.size))
        DEe = decreasing_exponential(abss, σe)
        u_mirror = np.concatenate([np.flip(u).ravel()[:mr_lm],
                                   u.ravel(), np.flip(u).ravel()[1:]])
        Ke = dx*np.convolve(DEe, u_mirror, mode='same')[mr_lm:2*mr_lm+1]\
            .reshape(self.size, 1)

        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * v - θe)))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * v - θi))\
            .reshape(v.shape)

        return concatenate((du, dv)).ravel()


class WCDecExpTravelNonLocal1D(WCKernel):
    """Simulating traveling wave in non-localized inhibition neural system"""
    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        """Check of the linear algebra solution"""
        # Model param
        u, v = split(inp.reshape(2*self.size, 1), 2)
        F = self.F
        A, (θe, θi), (τe, τi), η, (σe, σi) = self.param.derivative_tuple

        # Space param
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
        du = 1/(η*τe)*(-u + F((A[0, 0] * Ke - A[0, 1] * Ki - θe)))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * Ki - θi))\
            .reshape(v.shape)

        return concatenate((du, dv)).ravel()


# TODO: Actually get this to work
class WCDecExpMondronomy(WCKernel):
    """System for determining the Mondronomy Matrix"""
    def __init__(self, inp: Tuple[ndarray], param: Param):
        super().__init__(inp, param)
        self._simple = True

    @property
    def initial_inp_matrix(self):
        """Monkey patch to fix ragged input for X matrix with u and v"""
        pass

    @property
    def initial_inp_ravel(self):
        return self._init_inp

    def update(self, t: Tuple[int], inp: ndarray) -> ndarray:
        X = inp[0:4].reshape(2, 2)
        u, v = inp[4:]
        F = self.F
        A, (θe, θi), (τe, τi), η, _, _ = self.param.derivative_tuple
        AA = calc_AA(u, v, self.param)

        # # Iterate U an V values
        du = 1/(η*τe)*(-u + F((A[0, 0] * u - A[0, 1] * v - θe)))
        dv = 1/(η*τi)*(-v + F(A[1, 0] * u - A[1, 1] * v - θi))

        # Calculate X values
        dX = (AA @ X).ravel()

        # return dX.ravel()
        return concatenate((dX.ravel(), np.array((du, dv)))).ravel()


class WCDecExpTravelNonLocal2D(WCKernel):
    """Simulating traveling wave in 2D in a non-localized inhibition neural
    system"""

    def __init__(self, inp: Tuple[ndarray], param: Param):
        super().__init__(inp, param)
        self._make_kernels()
        self._simple = True

    def _make_kernels(self):
        x_lm = 24  # Found heuristically. No rational for limit from literature
        self.dx = x_lm/self.size
        rang = np.linspace(-x_lm/2, x_lm/2, x_lm)
        xx, yy = np.meshgrid(rang, rang)
        σe, σi = self.σ

        dist_2norm = la.norm(np.stack((xx, yy)), axis=0)
        self.DEe = decreasing_exponential(dist_2norm, σe)
        self.DEi = decreasing_exponential(dist_2norm, σi)

    def update(self, t: Tuple[int], inp: ndarray,
               timer=None, boundary='symm') -> ndarray:
        """Check of the linear algebra solution"""
        # Model param
        u, v = inp.reshape(2, self.size, self.size)
        F = self.F
        A, (θe, θi), (τe, τi), η, _ = self.param.derivative_tuple

        # Space param
        # print(DEe.shape, u.shape, DEi.shape, v.shape,)
        Ke = self.dx*convolve2d(u, self.DEe, mode='same', boundary=boundary)
        Ki = self.dx*convolve2d(v, self.DEi, mode='same', boundary=boundary)

        # ODE equations
        du = 1/(η*τe)*(-u + F(A[0, 0] * Ke - A[0, 1] * Ki - θe))\
            .reshape(u.shape)

        dv = 1/(η*τi)*(-v + F(A[1, 0] * Ke - A[1, 1] * Ki - θi))\
            .reshape(v.shape)

        if timer:
            timer(t)

        return concatenate((du, dv)).ravel()

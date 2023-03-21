import numpy as np
from numpy import stack
from scipy.integrate import solve_ivp

# Class Niceities
from dataclasses import dataclass
from abc import ABC, abstractproperty, abstractmethod

from ..kernels.kernels import (make_K_2_populations,
                               decreasing_exponential as dec_exp)

# Typing
from numpy import ndarray
from typing import Tuple, List, Callable, NewType


@dataclass
class WCKernelParam:
    A: ndarray  # [[a_ee, a_ei],[a_ie, a_ii]]
    Θ: ndarray  # [Θe, Θi]
    τ: ndarray  # [τe==1, 1/τi]
    size: int
    F: Callable


Param = NewType("Param", WCKernelParam)


class SolvedSystem:
    def __init__(self, solved, ws, dws):
        self._solved = solved
        self._ws = ws
        self._dws = dws

    def __getitem__(self, key) -> ndarray:
        return np.array([self._solved.t[key], self._ws[key], self._dws[key]])

    def __len__(self):
        return self._solved.t.size

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    def u_tuples(self) -> List[Tuple[ndarray]]:
        return [(t, w[0], dw[0]) for (t, w, dw) in self]

    def v_tuples(self) -> List[Tuple[ndarray]]:
        return [(t, w[1], dw[1]) for (t, w, dw) in self]

    @property
    def t(self):
        return self._solved.t

    @property
    def us(self):
        return np.array([w[0] for w in self._ws])

    @property
    def vs(self):
        return np.array([w[1] for w in self._ws])

    @property
    def dus(self):
        return np.array([dw[0] for dw in self._dws])

    @property
    def dvs(self):
        return np.array([dw[1] for dw in self._dws])

    def u_tuples(self) -> List[Tuple[ndarray]]:
        return [(t, w[0], dw[0]) for (t, w, dw) in self]

    def v_tuples(self) -> List[Tuple[ndarray]]:
        return [(t, w[1], dw[1]) for (t, w, dw) in self]


class WCKernel(ABC):
    def __init__(self, u, v, param: WCKernelParam):
        self._param = param
        self._init_u = u
        self._init_v = v
        self._init_w = stack([u, v])
        self._make_grid()

    def update(self, t, w) -> ndarray:
        w = w.reshape(2, self.size)
        A = self.A * np.array([[1, -1], [1, -1]])  # To subtract ends
        F = self.F
        Θ = self.θ
        τ = self.τ
        KK = self.kernel_grid

        x = ((F((A @ (KK @ w.T)[0].T) - Θ) - w) * τ)
        return x.ravel()

    def __call__(self, time, **kwargs) -> SolvedSystem:
        slv = solve_ivp(self.update, time, self.initial_w.ravel(), **kwargs)
        ln = slv.t.size
        ws = np.array([slv.y[:, i].reshape(2, self.size) for i in range(ln)])
        dws = np.array([self.update(t, w).reshape(2, self.size) for t, w in zip(slv.t, ws)])

        return SolvedSystem(slv, ws, dws)

    @abstractmethod
    def _make_grid(self):
        pass

    @abstractproperty
    def kernel_grid(self):
        pass

    @property
    def grid(self):
        return self._grid

    @property
    def initial_u(self):
        return self._init_u

    @property
    def initial_v(self):
        return self._init_v

    @property
    def initial_w(self):
        return self._init_w

    @property
    def A(self):
        return self._param.A

    @property
    def θ(self):
        return self._param.Θ.reshape([2, 1])

    @property
    def τ(self):
        return self._param.τ

    @property
    def F(self):
        return self._param.F

    @property
    def size(self):
        return self._param.size


class WCDecExp(WCKernel):
    def __init__(self, u: ndarray, v: ndarray, param: Param, σ: ndarray):
        super().__init__(u, v, param)
        self._σ = σ
        self._set_kernel_func()
        self._kernel_grid = None

    def _set_kernel_func(self):
        self._kernel_func = make_K_2_populations(dec_exp, self.σ)

    @property
    def σ(self):
        return self._σ

    @σ.setter
    def σ(self, other: ndarray):
        self._σ = other
        self._set_kernel_func()
        self._kernel_grid = None

    @property
    def kernel_func(self):
        return self._kernel_func

    @property
    def kernel_grid(self):
        if self._kernel_grid is not None:
            return self._kernel_grid
        self._kernel_grid = self.kernel_func(self.grid)
        return self._kernel_grid

    @abstractmethod
    def _make_grid(self):
        pass

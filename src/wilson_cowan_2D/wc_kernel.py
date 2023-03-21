import numpy as np
from numpy import stack
from scipy.optimize import solve_ivp

# Class Niceities
from dataclasses import dataclass
from abc import ABC, abstractproperty

# Typing
from numpy import ndarray
from typing import Tuple, List, Callable


@dataclass
class WCKernelParam:
    A: ndarray  # [[a_ee, a_ei],[a_ie, a_ii]]
    Θ: ndarray  # [Θe, Θi]
    τ: ndarray  # [τe==1, 1/τi]
    size: int
    F: Callable


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

    def us_as_list(self) -> List[Tuple[ndarray]]:
        return [(t, w[0], dw[0]) for (t, w, dw) in self]

    def vs_as_list(self) -> List[Tuple[ndarray]]:
        return [(t, w[1], dw[1]) for (t, w, dw) in self]


class WCKernel(ABC):
    def __init__(self, u, v, param: WCKernelParam):
        self._param = param
        self._init_u = u
        self._init_v = v
        self.init_w = stack([u, v])

    def update(self, t, w) -> ndarray:
        w = w.reshape(2, self.param.size)
        A = self.param.A * np.array([[1, -1], [1, -1]])  # To subtract ends
        F = self.param.F
        Θ = self.param.θ
        τ = self.param.τ
        KK = self.kernelgrid

        x = ((F((A @ (KK @ w.T)[0].T) - Θ) - w) * τ)
        return x.ravel()

    def __call__(self, time, **kwargs) -> SolvedSystem:
        slv = solve_ivp(self.update, time, self.init_w.ravel(), **kwargs)
        size = slv.t.size
        ws = np.array([slv.y[:, i].reshape(2, size) for i in range(len(size))])
        dws = np.array([self.update(w) for w in ws])

        return SolvedSystem(slv, ws, dws)

    @abstractproperty
    def kernelgrid(self):
        pass

    @property
    def inital_u(self):
        return self._init_u

    @property
    def inital_v(self):
        return self._init_v

    @property
    def A(self):
        return self._param.A

    @property
    def Θ(self):
        return self._param.Θ

    @property
    def τ(self):
        return self._param.τ

    @property
    def F(self):
        return self._param.F

    @property
    def size(self):
        return self._param.size

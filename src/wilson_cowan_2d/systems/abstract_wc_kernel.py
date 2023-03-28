from numpy import concatenate, split
from scipy.integrate import solve_ivp

# Class Niceities
from typing import List
from dataclasses import dataclass
from abc import ABC, abstractproperty, abstractmethod

from .nulclines import nulclines_and_crosspoints
from ..kernels.kernels import (make_K_2_populations,
                               decreasing_exponential as dec_exp)

# Typing
from numpy import ndarray
from typing import Tuple, Callable, NewType


@dataclass
class WCKernelParam:
    A: ndarray  # [[a_ee, a_ei],[a_ie, a_ii]]
    Θ: ndarray  # [Θe, Θi]
    τ: ndarray  # [τe==1, 1/τi]
    β: float
    η: float
    size: int


Param = NewType("Param", WCKernelParam)


class SolvedSystem:
    def __init__(self, solved,
                 inps: List[ndarray], dinps: List[ndarray]):
        self._solved = solved
        self._inps = inps
        self._dinps = dinps

    def __getitem__(self, key: int) -> Tuple[float, Tuple[float]]:
        return (
            self._solved.t[key],
            tuple(i for i in concatenate(self._inps, axis=1))[key],
            tuple(di[0] for di in self._dinps[key])
        )

    def __len__(self):
        return self._solved.t.size

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    @property
    def t(self):
        return self._solved.t


class WCKernel(ABC):
    def __init__(self, inp: Tuple[ndarray], param: WCKernelParam):
        self._param = param
        self._init_inp = inp
        self._num_vars = len(inp)
        self._make_grid()

    def __call__(self, time: Tuple[float], **kwargs) -> SolvedSystem:
        slv = solve_ivp(self.update, time, self.initial_inp_ravel, **kwargs)
        inps = split(slv.y.T, self.num_vars, axis=1)
        dinps = [
            split(self.update(t, inp), self.num_vars)
            for t, inp in zip(slv.t, slv.y.T)
        ]
        if not slv.success:
            print("Solver Didn't finish with message:", slv.message)

        return self._get_solved_system(slv, inps, dinps)

    @abstractmethod
    def update(self, t, inp) -> ndarray:
        pass

    @abstractproperty
    def _get_solved_system(self) -> SolvedSystem:
        pass

    @abstractmethod
    def _make_grid(self):
        pass

    @abstractproperty
    def kernel_grid(self):
        pass

    @property
    def initial_inp(self) -> Tuple[ndarray]:
        return self._init_inp

    @property
    def initial_inp_matrix(self) -> ndarray:
        if self.initial_inp[0].size == 1:
            return concatenate(self.initial_inp)

        return concatenate(self.initial_inp, axis=1).T

    @property
    def initial_inp_ravel(self) -> ndarray:
        return self.initial_inp_matrix.ravel()

    @property
    def initial_inp_vecs(self) -> List[ndarray]:
        return split(self.initial_inp_ravel, self.num_vars)

    @property
    def grid(self) -> ndarray:
        return self._grid

    @property
    def A(self) -> ndarray:
        return self._param.A

    @property
    def θ(self) -> ndarray:
        return self._param.Θ.reshape([2, 1])

    @property
    def τ(self) -> float:
        return self._param.τ

    @property
    def η(self) -> float:
        return self._param.η

    @property
    def F(self) -> Callable:
        return self._param.F

    @property
    def size(self) -> int:
        return self._param.size

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def nulclines_and_crosspoints(self, interp_prec=1e-3, fit_points=250):
        return nulclines_and_crosspoints(self.params, interp_prec, fit_points)


class WCDecExp(WCKernel):
    def __init__(self, inp: Tuple[ndarray], param: Param, σ: ndarray):
        super().__init__(inp, param)
        self._σ = σ
        self._set_kernel_func()
        self._kernel_grid = None

    def _set_kernel_func(self):
        self._kernel_func = make_K_2_populations(dec_exp, self.σ)

    @property
    def σ(self) -> ndarray:
        return self._σ

    @σ.setter
    def σ(self, other: ndarray):
        self._σ = other
        self._set_kernel_func()
        self._kernel_grid = None

    @property
    def kernel_func(self) -> Callable:
        return self._kernel_func

    @property
    def kernel_grid(self) -> ndarray:
        if self._kernel_grid is not None:
            return self._kernel_grid
        self._kernel_grid = self.kernel_func(self.grid)
        return self._kernel_grid

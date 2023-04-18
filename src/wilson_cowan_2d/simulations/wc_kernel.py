from numpy import concatenate, split, array
from scipy.integrate import solve_ivp

# Class Niceities
from abc import ABC, abstractmethod

# Typing
from numpy import ndarray
from typing import Tuple, Callable, List
from . import SimResult, Param
from .solved_system import SolvedSystem


class WCKernel(ABC):
    def __init__(self, inp: Tuple[ndarray], param: Param):
        self._param = param
        self._init_inp = inp
        self._num_vars = len(inp)
        self._simple = False

    def __call__(self, time: Tuple[float], simple=False, **kwargs) -> SimResult:
        simple = simple or self._simple
        slv = solve_ivp(self.update, time, self.initial_inp_ravel, **kwargs)
        if simple: return slv

        inps = split(slv.y.T, self.num_vars, axis=1)
        dinps = [
            split(self.update(t, inp), self.num_vars)
            for t, inp in zip(slv.t, slv.y.T)
        ]
        if not slv.success:
            print("Solver Didn't finish with message:", slv.message)

        return self._get_solved_system(slv, inps, dinps)

    @abstractmethod
    def update(self, t: float, inp: ndarray) -> ndarray:
        pass

    @property
    def _get_solved_system(self) -> SimResult:
        return SolvedSystem

    @property
    def initial_inp(self) -> Tuple[ndarray]:
        return self._init_inp

    @property
    def initial_inp_matrix(self) -> ndarray:
        if self.initial_inp[0].size == 1:
            # return concatenate(self.initial_inp)
            return array(self.initial_inp)

        return concatenate(self.initial_inp, axis=1).T

    @property
    def initial_inp_ravel(self) -> ndarray:
        return self.initial_inp_matrix.ravel()

    @property
    def initial_inp_vecs(self) -> List[ndarray]:
        return split(self.initial_inp_ravel, self.num_vars)

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
    def σ(self) -> ndarray:
        return self._param.σ

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def param(self) -> Param:
        return self._param

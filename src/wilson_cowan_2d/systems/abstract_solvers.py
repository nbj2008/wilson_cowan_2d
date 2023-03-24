import numpy as np
from abc import ABC, abstractmethod


class RK4Out(ABC):
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def insert(self):
        pass


class RK4(ABC):
    def __init__(self, derivative, h=0.01):
        self._derivative = derivative
        self._h = h

    @abstractmethod
    def _out_class(self) -> RK4Out:
        pass

    def __call__(self, init_inp, sys_range, h=None, **kwargs):
        if h:
            self.h = h

        self._clear_ks()
        ret = self._out_class(int(np.ceil(sys_range[1] / self.h)))
        inp = init_inp

        # Inital Values
        ret.insert(0, sys_range[0], inp, (None, None))  # No Derivaties yet

        # Solve System
        for ix, r in enumerate(
                np.arange(sys_range[0]+self.h, sys_range[1], self.h)):

            inp = self._solve_step(r, inp, **kwargs)
            ret.insert(ix+1, r, inp, self._k1)

        self._calc_k1(r, inp, **kwargs)
        return ret

    def _solve_step(self, r, inp, **kwargs):
        # Make sure theres a clean start
        self._clear_ks()

        for k_func in self._calc_iter:
            k_func(r, inp, **kwargs)

        return tuple(y + 1/6*(k1 + 2*k2 + 2*k3 + k4)*self.h
                     for (y, k1, k2, k3, k4) in
                     zip(inp, self._k1, self._k2, self._k3, self._k4))

    @property
    def _calc_iter(self):
        return (self._calc_k1, self._calc_k2, self._calc_k3, self._calc_k4)

    def _calc_k1(self, r, inp, **kwargs):
        self._k1 = self._derivative(r, inp, **kwargs)

    def _calc_k2(self, r, inp, **kwargs):
        self._k2 = self._derivative(
            r + self.h/2, tuple(self.h/2*k + y for k, y in zip(self._k1, inp)),
            **kwargs)

    def _calc_k3(self, r, inp, **kwargs):
        self._k3 = self._derivative(
            r + self.h/2, tuple(self.h/2*k + y for k, y in zip(self._k2, inp)),
            **kwargs)

    def _calc_k4(self, r, inp, **kwargs):
        self._k4 = self._derivative(
            r + self.h, tuple(self.h*k + y for k, y in zip(self._k3, inp)),
            **kwargs)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, other):
        assert other > 0, "Change must move forward!"
        self._h = other

    def _clear_ks(self):
        self._k1 = None
        self._k2 = None
        self._k3 = None
        self._k4 = None

import numpy as np
from .abstract_solvers import RK4Out, RK4
from ..kernels.grids import Dist1DGrid


class UVRK4Out(RK4Out):
    def __init__(self, size):
        super().__init__(size)
        self._t = [None]*size
        self._u = [None]*size
        self._v = [None]*size
        self._du = [None]*size
        self._dv = [None]*size

    def __getitem__(self, key):
        return (self._t[key], self._u[key], self._v[key],
                self._du[key], self._dv[key])

    def insert(self, key, t, inp, dinp, *_):
        self._t[key] = t
        self._u[key] = inp[0]
        self._v[key] = inp[1]
        self._du[key-1] = dinp[0]
        self._dv[key-1] = dinp[1]

    @property
    def t(self):
        return self._t

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    @property
    def du(self):
        return self._du

    @property
    def dv(self):
        return self._dv


class WZRK4Out(RK4Out):
    def __init__(self, size, grid_dim):
        super().__init__(size)
        self._grid_dim = grid_dim
        self._dist_grid = Dist1DGrid(self.grid_dim)
        self._x = [None] * size
        self._w = [None] * size
        self._z = [None] * size

    def __getitem__(self, key):
        return (self._x[key], self._w[key], self._z[key])

    def insert(self, key, t, inp, *_):
        self._x[key] = t
        self._w[key] = inp[0]
        self._z[key] = inp[1]

    @property
    def grid_dim(self):
        return self._grid_dim

    @property
    def dist_grid(self):
        return self._dist_grid.grid

    @property
    def x(self):
        return self._x

    @property
    def whole_indexs(self):
        return [ix for ix in range(len(self))
                if np.floor(self.x[ix]) == self.x[ix]]

    @property
    def w(self):
        return self._w

    @property
    def z(self):
        return self._z

    @property
    def w_vec(self):
        """Returns 1D Column matrix of w"""
        out = np.concatenate(
            np.array([self.w[ix] for ix in self.whole_indexs]), axis=1)
        out_mirrored = np.concatenate([out[:, :0:-1], out], axis=1)
        output = out_mirrored[self.dist_grid[0],
                              self.dist_grid + (self.grid_dim - 1)]

        return output.sum(axis=1).reshape((self.grid_dim, 1))

    @property
    def z_vec(self):
        """Returns 1D Column matrix of z"""
        out = np.concatenate(
            np.array([self.z[ix] for ix in self.whole_indexs]), axis=1)
        out_mirrored = np.concatenate([out[:, :0:-1], out], axis=1)
        output = out_mirrored[self.dist_grid[0],
                              self.dist_grid + (self.grid_dim - 1)]

        return output.sum(axis=1).reshape((self.grid_dim, 1))


class UVRK4(RK4):
    def _out_class(self, size):
        return UVRK4Out(size)


class WZRK4(RK4):
    def __init__(self, derivative, grid_dim, h=0.01):
        super().__init__(derivative, h)
        self.grid_dim = grid_dim

    def _out_class(self, size):
        return WZRK4Out(size, self.grid_dim)

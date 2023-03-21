from numpy import concatenate, repeat, stack, arange, abs as nabs, mgrid, ones
from scipy.linalg import norm
from itertools import product
from abc import ABC, abstractmethod


class Grid(ABC):
    def __init__(self, size):
        self._size = size
        self._grid = None  # Initialize Grid Here

    @abstractmethod
    def __getitem__(self, key):
        pass

    @property
    def grid(self):
        return self._grid

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._grid.shape


class UnifGrid(Grid):
    def __init__(self, size):
        super().__init__(size)
        self.grid = ones(size)

    def __getitem__(self):
        return 1


class Dist2DGrid(Grid):
    def __init__(self, size):
        super().__init__(size)
        self._grid = make_2d_dist_grid(size)

    def __getitem__(self, key):
        """Indexer for Grid. 0-> x-axis, 1->y-axis"""
        assert key[0] < self.size, f"X value must be less than {self.size}"
        assert key[1] < self.size, f"Y value must be less than {self.size}"
        return self._grid[key[1]*self.size + key[0]]


class Dist1DGrid(Grid):
    def __init__(self, size):
        super().__init__(size)
        self._grid = make_1d_dist_grid(size)

    def __getitem__(self, key):
        assert key < self.size, f"Index must be less than {self.size}"
        return self._grid[key]


def make_2d_dist_grid(size):
    rsize = range(size)
    x_row = concatenate([repeat(i, size) for i in rsize])
    y_row = concatenate([arange(size)]*size)
    return norm(
        stack(
            [stack([y_row-y, x_row-x]) for x, y in product(rsize, rsize)]
        ), axis=1).reshape(size**2, size, size)


def make_1d_dist_grid(size):
    acs, dwn = mgrid[:size, :size]
    return nabs((acs-dwn).reshape(size, size))

from numpy import array, ndarray, concatenate

# Typing
from typing import List, Tuple, NewType


class SolvedSystem:
    def __init__(self, solved, inps: List[ndarray], dinps: List[ndarray]):
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
    def solved(self):
        return self._solved

    @property
    def t(self):
        return self._solved.t

    @property
    def u(self) -> ndarray:
        return self._inps[0]

    @property
    def v(self) -> ndarray:
        return self._inps[1]

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])


SimResult = NewType("SimResult", SolvedSystem)

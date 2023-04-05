from numpy import array, ndarray
from .abstract_wc_kernel import SolvedSystem


class UnifSolvedSystem(SolvedSystem):
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


class Local1DSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return self._inps[0].T

    @property
    def v(self) -> ndarray:
        return self._inps[1].T

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])


class NonLocal1DSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return self._inps[0].T

    @property
    def v(self) -> ndarray:
        return self._inps[1].T

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])

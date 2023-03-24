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
    def w(self) -> ndarray:
        return self._inps[2].T

    @property
    def z(self) -> ndarray:
        return self._inps[3].T

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])

    @property
    def dw(self) -> ndarray:
        return array([dinp[2] for dinp in self._dinps])

    @property
    def dz(self) -> ndarray:
        return array([dinp[3] for dinp in self._dinps])


class NonLocal1DSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return self._inps[0].T

    @property
    def v(self) -> ndarray:
        return self._inps[1].T

    @property
    def w(self) -> ndarray:
        return self._inps[2].T

    @property
    def z(self) -> ndarray:
        return self._inps[3].T

    @property
    def q(self) -> ndarray:
        return self._inps[4].T

    @property
    def p(self) -> ndarray:
        return self._inps[5].T

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])

    @property
    def dw(self) -> ndarray:
        return array([dinp[2] for dinp in self._dinps])

    @property
    def dz(self) -> ndarray:
        return array([dinp[3] for dinp in self._dinps])

    @property
    def dq(self) -> ndarray:
        return array([dinp[4] for dinp in self._dinps])

    @property
    def dp(self) -> ndarray:
        return array([dinp[5] for dinp in self._dinps])

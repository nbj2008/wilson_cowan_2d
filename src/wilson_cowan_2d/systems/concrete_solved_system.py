from numpy import array, ndarray
from .abstract_wc_kernel import SolvedSystem


class UnifSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return array([inp[0] for inp in self._inps])

    @property
    def v(self) -> ndarray:
        return array([inp[1] for inp in self._inps])

    @property
    def du(self) -> ndarray:
        return array([dinp[0] for dinp in self._dinps])

    @property
    def dv(self) -> ndarray:
        return array([dinp[1] for dinp in self._dinps])


class Local1DSolvedSystem(SolvedSystem):
    @property
    def u(self) -> ndarray:
        return array([inp[0] for inp in self._inps])

    @property
    def v(self) -> ndarray:
        return array([inp[1] for inp in self._inps])

    @property
    def w(self) -> ndarray:
        return array([inp[2] for inp in self._inps])

    @property
    def z(self) -> ndarray:
        return array([inp[3] for inp in self._inps])

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
        return array([inp[0] for inp in self._inps])

    @property
    def v(self) -> ndarray:
        return array([inp[1] for inp in self._inps])

    @property
    def w(self) -> ndarray:
        return array([inp[2] for inp in self._inps])

    @property
    def z(self) -> ndarray:
        return array([inp[3] for inp in self._inps])

    @property
    def q(self) -> ndarray:
        return array([inp[4] for inp in self._inps])

    @property
    def p(self) -> ndarray:
        return array([inp[5] for inp in self._inps])

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

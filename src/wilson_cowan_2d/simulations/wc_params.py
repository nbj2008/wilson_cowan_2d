from dataclasses import dataclass
from numpy import ndarray, ones, array
from .nonlinear_functions import sigmoid, ssn_power_law

# Typing
from typing import NewType


@dataclass
class WCKernelParam:
    A: ndarray  # [[a_ee, a_ei],[a_ie, a_ii]]
    Θ: ndarray  # [Θe, Θi]
    τ: ndarray  # [τe==1, τi]
    β: float
    η: float
    size: int
    σ: ndarray = ones(2)  # [σe ==1, σi]

    @property
    def derivative_tuple(self):
        return (self.A, self.Θ, self.τ, self.η, self.σ)


Param = NewType("Param", WCKernelParam)
_default_ssn_values = {"A": array([[1.5, 1], [10, 1]]),
                       "Θ": array([-5, -0.01]), "η": 1, "σ": ones(2), "β": 50}


class DefaultParams(WCKernelParam):
    """Default set of Parameters used in Harris 2018"""
    def __init__(self, τ: ndarray, η: float, size: int, σ: ndarray = ones(2)):
        super().__init__(
             A=array([[1, 1.5], [1, 0.25]]),
             Θ=array([0.125, 0.4]),
             β=50,
             τ=τ, size=size,
             η=η, σ=σ)

    def F(self, other):
        return sigmoid(other, self.β)  # 1/(1 + nexp(-self.β * other))


class SSNDefaultParams(WCKernelParam):
    """Default set of Parameters used in Harris 2018"""

    def __init__(self, τ: ndarray, n: float, size: int, k: float = 1, **kwargs):
        # inp = _default_ssn_values | kwargs  # Gives defaults unless in kwargs
        inp = _default_ssn_values.copy()
        for key, val in kwargs.items():
            inp[key] = val

        super().__init__(τ=τ, size=size, **inp)
        self.n = n
        self.k = k

    def F(self, other):
        return ssn_power_law(other, self.n, self.k)


class MondronomyParams(WCKernelParam):
    """Parameters for calculating Mondronomy Matrices"""
    def __init__(self, A: ndarray, Θ: ndarray, τ: float, β: float,
                 η: float, size: int, σ: ndarray, ω: float):
        super().__init__(A, Θ, τ, β, η, size, σ)
        self.ω = ω

    @classmethod
    def default(cls, τ, η, size, σ, ω):
        return cls.from_system_params(DefaultParams(τ, η, size, σ), ω)

    @classmethod
    def from_system_params(cls, o: Param, ω: float):
        return MondronomyParams(o.A, o.Θ, o.τ, o.β, o.η, o.size, o.σ, ω)

    @property
    def derivative_tuple(self):
        return (self.A, self.Θ, self.τ, self.η, self.σ, self.ω)

    def F(self, other):
        return sigmoid(other, self.β)  # 1/(1 + nexp(-self.β * other))


class SSNMondronomyParams(SSNDefaultParams):
    """Parameters for calculating Mondronomy Matrices"""
    def __init__(self, ω, **kwargs):
        super().__init__(**kwargs)
        self.ω = ω

    @classmethod
    def from_system_params(cls, o: Param, ω: float):
        return MondronomyParams(o.A, o.Θ, o.τ, o.β, o.η, o.size, o.σ, ω)

    @property
    def derivative_tuple(self):
        return (self.A, self.Θ, self.τ, self.η, self.σ, self.ω)

    def F(self, other):
        return ssn_power_law(other, self.n, self.k)

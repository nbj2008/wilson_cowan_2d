import numpy as np
from scipy.linalg import svd
from scipy.spatial import ConvexHull

from typing import NewType, List, Tuple
from ..simulations import SimResult, Param

fr = NewType("fr", float)

_dflt_range = np.linspace(0, 100, 1_000)


def calc_D0(u_bar: fr,  v_bar: fr,  param: Param) -> np.ndarray:
    """Calculates the Determinate with ω = 0 from
    the Linear Stablility Analysis in Harris 2018
    """

    bs = calc_bs(param,  u_bar,  v_bar)
    return (1 - bs[0, 0])*(1 + bs[1, 1]) + bs[0, 1]*bs[1, 0]


def calc_Dω(param: Param, equib_point: Tuple[fr], σ: float,
            point_range: np.ndarray = _dflt_range) -> np.ndarray:
    """Calculates the Determinate for given ω from
    the Linear Stablility Analysis in Harris 2018
    """

    bs = calc_bs(param, equib_point[0], equib_point[1])
    e_ft = analytic_FT_decexp(σ[0], point_range)
    i_ft = analytic_FT_decexp(σ[1], point_range)

    return _Dω_equation(bs, e_ft, i_ft)


def calc_Dω_range(param: Param, equib_point: Tuple[fr], σe: float, σi_range: np.ndarray,
                  point_range: np.ndarray = _dflt_range) -> List[np.ndarray]:
    """Calculates the Determinate over range of ωs from
    the Linear Stablility Analysis in Harris 2018
    """

    bs = calc_bs(param, equib_point[0], equib_point[1])
    e_ft = analytic_FT_decexp(σe, point_range)

    return [_Dω_equation(bs, e_ft, analytic_FT_decexp(σi, point_range))
            for σi in σi_range]


def _Dω_equation(bs: np.ndarray, eft: np.ndarray, ift: np.ndarray) -> np.ndarray:
    return (1 - bs[0, 0]*eft)*(1 + bs[1, 1]*ift) + bs[0, 1]*bs[1, 0]*eft*ift


def calc_bs(param: Param,  u_bar: fr,  v_bar: fr) -> np.ndarray:
    """Calculates the values of b from equation 5 in Harris 2018"""

    if u_bar.size == 1:
        init = derv_F(param, u_bar, v_bar).reshape(2, 1)
        return param.A * init
    else:
        init = [x.reshape(2, 1) for x in derv_F(param,  u_bar,  v_bar)]
        return np.array([param.A * x for x in init])
    # return param.A * derv_F(param,  u_bar,  v_bar).reshape(2, u_bar.size)


def derv_F(param: Param,  u_bar: fr,  v_bar: fr) -> np.ndarray:
    """Calculates the derivative of the Fourier Transform of the F function"""
    ss = ss_F(param,  u_bar,  v_bar)
    return param.β * ss*(1 - ss)


def ss_F(param: Param,  u_bar: fr,  v_bar: fr) -> np.ndarray:
    """Provides the nonlinear function of equation 3 in Harris 2018"""

    return param.F(param.A[:,  0]*u_bar - param.A[:, 1]*v_bar - param.Θ)


def analytic_FT_decexp(σ: float,  ω: float) -> float:
    """Analytic equation of the Fourier Transform
    of the decreasing exponential"""

    return 1/(1 + (σ * ω)**2)


def fit_ellipse_periodic_orbit(res: SimResult, N: int = 1_000,
                               period: float = 2*np.pi) -> np.ndarray:
    """Fits an ellipse to the loose oribts about the E3 fixed point.
    Don't actually use this """

    # Set Fit Parms
    N = 1_000
    u_mean, v_mean = res.u.mean(), res.v.mean()
    t = np.linspace(0, period, N)

    # Extract Data
    eu, ev = res.u-u_mean, res.v-v_mean
    U, S, V = svd(np.stack((eu.ravel(), ev.ravel())))

    # Fit Ellipse
    unit_circle = np.vstack((np.cos(t), np.sin(t)))
    transform = np.sqrt(2/len(res)) * U.dot(np.diag(S))
    return transform.dot(unit_circle) + np.array([[u_mean], [v_mean]])


def fit_convex_hull(res):
    """Finds the convex hull of the simulated orbits around the E3 fixed point.
    Don't actually use this """

    points = np.concatenate((res.u, res.v), axis=1)
    hull = ConvexHull(points)
    return (points[hull.vertices, 0], points[hull.vertices, 1])


def calc_AA(u, v, param):
    """Calculates the coefficient matrix for periodic orbits from simulation.
    For determining the Mondronomy Matrix."""

    bs = calc_bs(param, u, v)
    Fe = analytic_FT_decexp(param.σ[0], param.ω)
    Fi = analytic_FT_decexp(param.σ[1], param.ω)
    τ = param.τ[1]/param.τ[0]

    return np.array([[-1 + bs[0, 0]*Fe, -bs[0, 1]*Fi],
                     [(bs[1, 0]*Fe)/τ, (-1 - bs[1, 1]*Fi)/τ]
                     ])

import numpy as np
import scipy.optimize as opt
from functools import partial

from typing import NewType, Tuple, List, Callable
from ..simulations import Param
fr = NewType("fr", float)


def calc_nulclines_crosspoints(params: Param, interp_prec: float = 0.001,
                               fit_points: int = 250,
                               t_rang: Tuple[float]=(0,1)) -> Tuple[float, float, np.ndarray]:
    """Produces the U and V nulclines as well as their crossing points"""
    uinterp, vinterp = calc_nulclines(params, interp_prec, fit_points, t_rang)
    cps = _find_cross_points(uinterp[0], uinterp[1], vinterp[1])
    return uinterp, vinterp, cps[:, 1:]


def calc_cross_points(params: Param, interp_prec: float = 1e-3,
                      fit_points: int = 250,
                      t_rang: Tuple[float]=(0,1)) -> np.ndarray:
    """Calculates the crossing points of the U and V nulclines"""
    uinterp, vinterp = calc_nulclines(params, interp_prec, fit_points, t_rang)
    return _find_cross_points(uinterp[0], uinterp[1], vinterp[1])


def calc_nulclines(params: Param, interp_prec: float = 1e-3,
                   fit_points: int = 250,
                   t_rang: Tuple[float]=(0,1)) -> Tuple[np.ndarray]:
    """Calculates the nulclines of the U and V firing rates"""
    e_func = partial(_e_min_func, params=params)
    i_func = partial(_i_min_func, params=params)

    us = np.linspace(t_rang[0], t_rang[1], fit_points)
    ves = _generate_fits(e_func, us)
    vis = _generate_fits(i_func, us)

    tt = np.arange(t_rang[0], t_rang[1], interp_prec)
    nucs = _interpolate_nulclines((us, ves), (us, vis), tt)
    return tuple(np.stack((tt, n)) for n in nucs)


def _generate_fits(func: Callable, rang: np.ndarray = np.linspace(0, 1, 250)) -> List[float]:
    return [opt.minimize(func, 0.5, args=(v), method='nelder-mead').x[0]
            for v in rang]


def _interpolate_nulclines(us: np.ndarray, vs: np.ndarray,
                           interp_range: np.ndarray = np.arange(0, 1, 1e-3)) -> List[np.ndarray]:
    return [np.interp(interp_range, ncs[0], ncs[1]) for ncs in (us, vs)]


def _find_cross_points(rang: np.ndarray, uinterp: fr, vinterp: fr) -> np.ndarray:
    rr = np.sign(uinterp - vinterp)
    cond = np.where(rr - np.roll(rr, 1) != 0)
    rx = rang[cond]
    vx = vinterp[cond]

    return np.stack([rx, vx])


def _e_min_func(v: fr, u: fr, params) -> np.ndarray:
    """From equation 3 in Harris 2018"""
    return np.abs(
        params.F(params.A[0, 0]*u - params.A[0, 1]*v - params.Θ[0]) - u)   + 1e-8*np.abs(v)


def _i_min_func(v, u, params) -> np.ndarray:
    """From equation 3 in Harris 2018"""
    return np.abs(
        params.F(params.A[1, 0]*u - params.A[1, 1]*v - params.Θ[1]) - v)   + 1e-8*np.abs(v)

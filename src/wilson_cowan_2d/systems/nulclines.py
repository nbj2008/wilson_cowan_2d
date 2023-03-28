import numpy as np
import scipy.optimize as opt
from functools import partial
from wilson_cowan_2d.systems.abstract_wc_kernel import Param


def calc_nulclines_crosspoints(params: Param, interp_prec=0.001, fit_points=250):
    uinterp, vinterp = calc_nulclines(params, interp_prec, fit_points)
    cps = _find_cross_points(uinterp[0], uinterp[1], vinterp[1])
    return uinterp, vinterp, cps


def calc_cross_points(params: Param, interp_prec=1e-3, fit_points=250):
    uinterp, vinterp = calc_nulclines(params, interp_prec, fit_points)
    return _find_cross_points(uinterp[0], uinterp[1], vinterp[1])


def calc_nulclines(params: Param, interp_prec=1e-3, fit_points=250):
    u_func = partial(_u_min_func, params=params)
    v_func = partial(_v_min_func, params=params)

    uus = vvs = np.linspace(0, 1, fit_points)
    uvs = _generate_fits(u_func, uus)
    vus = _generate_fits(v_func, vvs)

    tt = np.arange(0, 1, interp_prec)
    nucs = _interpolate_nulclines((uus, uvs), (vus, vvs), tt)
    return tuple(np.stack((tt, n)) for n in nucs)


def _generate_fits(func, rang=np.linspace(0, 1, 250)):
    return [opt.minimize(func, v, args=(v), method='nelder-mead').x[0]
            for v in rang]


def _interpolate_nulclines(us, vs, interp_range=np.arange(0, 1, 1e-3)):
    return [np.interp(interp_range, ncs[0], ncs[1]) for ncs in (us, vs)]


def _find_cross_points(rang, uinterp, vinterp):
    rr = np.sign(uinterp - vinterp)
    cond = np.where(rr - np.roll(rr, 1) != 0)
    rx = rang[cond]
    vx = vinterp[cond]

    return np.stack([rx, vx])


def _u_min_func(v, u, param: Param):
    return np.abs(param.F(param.A[0, 0]*u - param.A[0, 1]*v - param.Θ[0]) - u)


def _v_min_func(u, v, param: Param):
    return np.abs(param.F(param.A[1, 0]*u - param.A[1, 1]*v - param.Θ[1]) - v)

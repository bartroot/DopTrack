import sys
import logging
import autograd.numpy as np
import scipy.optimize as optimize

from .config import Config


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=Config().runtime['log_level'])


def tanh(xs, a, b, c, d):
    return -a*np.tanh((xs - d)/c) + b


def fit_tanh(xs, ys, dt):
    # initial guess
    a0 = (np.max(ys)-np.min(ys))/2
    b0 = np.mean(ys)
    c0 = 100/dt
    d0 = np.mean(xs)
    p0 = [a0, b0, c0, d0]

    ftol = 10**-8
    xtol = 10**-8
    max_nfev = 10000

    # prepare curve data
    # non-linear least squares
    fit_coeffs, covar = optimize.curve_fit(tanh, xs, ys, p0=p0, loss='soft_l1',
                                           method='trf', ftol=ftol, xtol=xtol, max_nfev=max_nfev)

    return fit_coeffs


def fourier4(x, a0, a1, a2, a3, a4, b1, b2, b3, b4, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p)


def fourier5(x, a0, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p) + \
            a5 * np.cos(5*x*p) + b5 * np.sin(5*x*p)


def fourier6(x, a0, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p) + \
            a5 * np.cos(5*x*p) + b5 * np.sin(5*x*p) + \
            a6 * np.cos(6*x*p) + b6 * np.sin(6*x*p)


def fourier3(x, a0, a1, a2, a3, b1, b2, b3, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p)


def p3(x, a0, a1, a2, a3, p):
    return a0 + a1 * (x - p) + a2 * (x - p)**2 + a3 * (x - p)**3


def p5(x, a0, a1, a2, a3, a4, a5, p):
    return a0 + a1 * (x - p) + a2 * (x - p)**2 + a3 * (x - p)**3 + \
           a4 * (x - p)**4 + a5 * (x - p)**5


def fit_residual(times, residual):
    max_nfev = 6000
    a = (times)/np.std(times)
    # non-linear least squares
    for func in [fourier6, fourier4, fourier3, fourier5, p3, p5]:
        try:
            fit_coeffs, covar = optimize.curve_fit(func, a, residual,
                                                   method='trf',
                                                   loss='soft_l1',
                                                   max_nfev=max_nfev)
            fit_coeffs[-1] = (fit_coeffs[-1]) / np.std(times)
            if np.abs(np.mean(times - func(times, *fit_coeffs))) > 10000:
                raise RuntimeError('Did not converge')
            logger.debug(f'Residual fitting converged using {func}')
            return func, fit_coeffs
        except RuntimeError:
            logger.warning(f'Fitting function {func} did not converge.')
            continue

    raise RuntimeError('None of the fitting functions converged.')

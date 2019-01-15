"""Functions for fitting data during time-frequency data extraction.

Classes
-------
- `FittingError` -- Exception thrown when fitting is unsuccesful.

Routines
--------
- `fit_tanh` -- Fit tanh function to time-frequency data.
- `fit_residual` -- Fit fourier function or polynomial to time-residual data.

"""
import logging
import autograd.numpy as np
import scipy.optimize as optimize


logger = logging.getLogger(__name__)


class FittingError(Exception):
    """Raised whenever the fitting algorithms fail in some way."""
    pass


def fit_tanh(times, frequencies, dt):
    """
    Extract frequency data from a spectrogram.

    Parameters
    ----------
    times : np.array
        The time series.
    frequencies : np.array
        The frequency series.
    dt : int or float
        The timestep of the spectrogram.
        Only used to give an initial guess of the coefficients.

    Returns
    -------
    np.array
        Array with the fitting coefficients of the tanh function.

    Raises
    ------
    FittingError
        If curve_fit fails in any way.
    """
    # initial guess
    a0 = (np.max(frequencies)-np.min(frequencies))/2
    b0 = np.mean(frequencies)
    c0 = 100/dt
    d0 = np.mean(times)
    p0 = [a0, b0, c0, d0]

    ftol = 10**-8
    xtol = 10**-8
    max_nfev = 10000

    # non-linear least squares
    try:
        fit_coeffs, covar = optimize.curve_fit(tanh, times, frequencies, p0=p0, loss='soft_l1',
                                               method='trf', ftol=ftol, xtol=xtol, max_nfev=max_nfev)
    except RuntimeError as e:
        raise FittingError(f'Fitting of tanh was unsuccessful: {e}')

    return fit_coeffs


def fit_residual(times, residual):
    """
    Extract frequency data from a spectrogram.

    Parameters
    ----------
    times : np.array
        Time series of data.
    residual : np.array
        Residual series of data.
    dt : int or float
        The timestep of the spectrogram.
        Only used to give an initial guess of the coefficients.

    Returns
    -------
    func
        The function used to fit the residual.
    np.array
        Array with the fitting coefficients of the residual function.

    Raises
    ------
    FittingError
        If none of the candidate residual functions converge during fitting.
    """

    max_nfev = 6000
    # TODO why is std normalization needed???
    a = (times)/np.std(times)

    for func in [fourier8, fourier7, fourier6, fourier5, fourier4, fourier3, fourier5, poly3, poly5]:
        try:
            fit_coeffs, covar = optimize.curve_fit(
                    func, a, residual,
                    method='lm',
                    maxfev=max_nfev)
            fit_coeffs[-1] = (fit_coeffs[-1]) / np.std(times)
            logger.debug(f'Residual fitting converged using {func.__name__}')
            return func, fit_coeffs
        except RuntimeError:
            logger.debug(f'Fitting function {func.__name__} did not converge')
            continue

    raise FittingError('None of the residual fitting functions converged')


def tanh(xs, a, b, c, d):
    return -a*np.tanh((xs - d)/c) + b


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


def fourier6(x, a0, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p) + \
            a5 * np.cos(5*x*p) + b5 * np.sin(5*x*p) + \
            a6 * np.cos(6*x*p) + b6 * np.sin(6*x*p)


def fourier7(x, a0, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p) + \
            a5 * np.cos(5*x*p) + b5 * np.sin(5*x*p) + \
            a6 * np.cos(6*x*p) + b6 * np.sin(6*x*p) + \
            a7 * np.cos(7*x*p) + b7 * np.sin(7*x*p)


def fourier8(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p) + \
            a5 * np.cos(5*x*p) + b5 * np.sin(5*x*p) + \
            a6 * np.cos(6*x*p) + b6 * np.sin(6*x*p) + \
            a7 * np.cos(7*x*p) + b7 * np.sin(7*x*p) + \
            a8 * np.cos(8*x*p) + b8 * np.sin(8*x*p)


def fourier9(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8, b9, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p) + \
            a5 * np.cos(5*x*p) + b5 * np.sin(5*x*p) + \
            a6 * np.cos(6*x*p) + b6 * np.sin(6*x*p) + \
            a7 * np.cos(7*x*p) + b7 * np.sin(7*x*p) + \
            a8 * np.cos(8*x*p) + b8 * np.sin(8*x*p) + \
            a9 * np.cos(9*x*p) + b9 * np.sin(9*x*p)


def fourier3(x, a0, a1, a2, a3, b1, b2, b3, p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p)


def poly3(x, a0, a1, a2, a3, p):
    return a0 + a1 * (x - p) + a2 * (x - p)**2 + a3 * (x - p)**3


def poly5(x, a0, a1, a2, a3, a4, a5, p):
    return a0 + a1 * (x - p) + a2 * (x - p)**2 + a3 * (x - p)**3 + \
           a4 * (x - p)**4 + a5 * (x - p)**5

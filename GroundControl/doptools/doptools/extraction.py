import sys
import os
import logging
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as optimize
from collections import Counter
from sklearn.cluster import DBSCAN

from .config import Config
from .model_doptrack import Spectrogram
from .masking import area_mask, fit_mask, time_mask
from .fitting import tanh, fit_tanh, fit_residual


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=Config().runtime['log_level'])


def extract_datapoints(dataid):

    logger.info(f'Extracting frequency data points from {dataid}')
    spectrogram = Spectrogram.load(dataid)
    image = spectrogram.image

    # First masking
    mask1 = area_mask(image, ori='freq', dt=spectrogram.dt)
    mask2 = area_mask(image, ori='time', dt=spectrogram.dt)
    mask = np.logical_and(mask1, mask2)

    # Apply mask and change zeros to the mean value of the data
    image_masked = image * mask

    # First fitting cycle
    sideband = 600
    threshs = sideband * np.array([2.5, 1.5, 0.5])
    xs, ys, vals, fitfunc = fitting_cycle(image_masked,
                                          spectrogram.dt,
                                          threshs=threshs,
                                          cluster_eps=25,
                                          plot=False)

    logger.info(f'1st fitting cycle completed.\n    Found {len(xs)} data points.')

    # Second masking
    mask1 = fit_mask(image, fitfunc)
    mask2 = time_mask(image, xs[0], xs[-1])
    mask = np.logical_and(mask1, mask2)
    image_masked = image_masked * mask

    # Second fitting cycle
    sideband = 600
    threshs = sideband * np.array([0.5])
    xs, ys, vals, fitfunc = fitting_cycle(image_masked,
                                          spectrogram.dt,
                                          threshs=threshs,
                                          cluster_eps=20,
                                          plot=False)

    logger.info(f'2nd fitting cycle completed.\n    Found {len(xs)} data points.')

    # Calculate bounds for root finding
    middle = xs[0] + (xs[-1] - xs[0]) / 2
    bound = (xs[-1] - xs[0]) / 8

    # Perform root finding to determine tca and fca
    tca = optimize.brentq(egrad(egrad(fitfunc)), middle - bound, middle + bound)
    fca = fitfunc(tca)

    logger.info(f'Additional values calculated:\n    tca: {tca}\n    fca: {fca}')

    plt.figure()
    plt.imshow(image_masked, clim=(0, 0.1), cmap='viridis', aspect='auto')
    plt.scatter(ys, xs, color='r')
    times = np.arange(image.shape[0])
    plt.plot(fitfunc(times), times, 'r')
    plt.show()

    return xs, ys, vals, tca, fca


def fitting_cycle(image, dt, threshs, cluster_eps, plot=False):

    xs, ys, vals = filter_datapoints(image, dt)

    # Repeatedly fit tangent curve to filtered data points
    for thresh in threshs:
        fit_coeffs = fit_tanh(xs, ys, dt)
        fit = tanh(xs, *fit_coeffs)
        xs, ys, vals = remove_outliers(xs, ys, vals, fit, thresh)
    fit = tanh(xs, *fit_coeffs)
    res = ys - fit

    # Find clusters of data points
    data = np.dstack((xs, res))[0]
    db = DBSCAN(eps=cluster_eps, min_samples=5).fit(data)
    data = np.column_stack(data)

    if plot:
        plt.figure()
        for i in set(db.labels_):
            plt.scatter(data[0][db.labels_ == i], data[1][db.labels_ == i], label=i)
        plt.legend()
        plt.show()

    # Find labels of all 'good' clusters with 10 or more data points
    counter = Counter(db.labels_)
    counter.pop(-1, None)
    labels = []
    for key, count in counter.most_common():
        if count < 10:
            break
        else:
            labels.append(key)

    # Remove data points not in one of the 'good' clusters
    bools = np.isin(db.labels_, labels)
    xs = data[0][bools]
    ys = ys[bools]
    vals = vals[bools]
    res = data[1][bools]

    # Create final fitting function
    func, coeffs = fit_residual(xs, res)

    def fitfunc(xs):
        """Calculates the fit for any value."""
        tanfunc = tanh(xs, *fit_coeffs)
        resfunc = func(xs, *coeffs)
        return tanfunc + resfunc

    return xs, ys, vals, fitfunc


def filter_datapoints(data, dt):

    noisefilter = signal.gaussian(int(14 / dt) + 1, 2.5)
    noisefilter = noisefilter / np.sum(noisefilter)

    mean = data.mean()

    vals = np.zeros(len(data))
    ys = np.zeros(len(data))

    for i, row in enumerate(data):
        row_norm = row - mean
        # Set all values less than 5 std from mean to zero
        row_norm[row_norm < 5 * np.std(row_norm)] = 0
        # Apply gaussian filter
        row_norm = np.convolve(row_norm, noisefilter, mode='same')

        vals[i] = np.max(row_norm)
        ys[i] = np.argmax(row_norm)

    xs = np.arange(len(ys))

    # Filter out data points where freq is zero
    bools = ys != 0
    xs = xs[bools]
    ys = ys[bools]
    vals = vals[bools]

    return xs, ys, vals


def remove_outliers(xs, ys, vals, mean, band):
    ys_ = ys[(ys > mean - band) & (ys < mean + band)]
    xs_ = xs[(ys > mean - band) & (ys < mean + band)]
    vals_ = vals[(ys > mean - band) & (ys < mean + band)]
    return xs_, ys_, vals_

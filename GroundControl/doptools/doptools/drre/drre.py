import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from collections import Counter
from sklearn.cluster import DBSCAN

from ..model_doptrack import Spectrogram
from .masks import area_mask
from .fitting import tanh, fit_tanh, fit_residual


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main(dataid):
    spectrogram = Spectrogram.load(dataid)
    image = spectrogram.image

    mask1 = area_mask(image, ori='freq', dt=spectrogram.dt)
    mask2 = area_mask(image, ori='time', dt=spectrogram.dt)
    mask = np.logical_and(mask1, mask2)

    # Apply mask and change zeros to the mean value of the data
    image_masked = image * mask + np.logical_not(mask) * image.mean()

    xs, ys, vals = filter_datapoints(image_masked, spectrogram.dt)

    sideband = 600

    for m in [2.5, 1.5, 0.5]:
        fit_coeffs = fit_tanh(xs, ys, spectrogram.dt)
        fit = tanh(xs, *fit_coeffs)
        xs, ys, vals = remove_outliers(xs, ys, vals, fit, m*sideband)
    fit = tanh(xs, *fit_coeffs)

    res = ys - fit

    np.row_stack
    data = np.dstack((xs, res))[0]
    db = DBSCAN(eps=25, min_samples=5).fit(data)
    data = np.column_stack(data)

    plt.figure()
    for i in set(db.labels_):
        plt.scatter(data[0][db.labels_ == i], data[1][db.labels_ == i], label=i)
    plt.legend()
    plt.show()

    cnt = Counter(db.labels_)
    j = cnt.most_common(1)[0][0]
    xs = data[0][db.labels_ == j]
    ys = ys[db.labels_ == j]
    res = data[1][db.labels_ == j]

    func, coeffs = fit_residual(xs, res)
    res_fit = func(xs, *coeffs)

    fit_total = tanh(xs, *fit_coeffs) + func(xs, *coeffs)

    plt.figure()
    plt.scatter(xs, res, color='r')
    plt.plot(xs, res_fit)
    plt.show()

    plt.figure()
    plt.imshow(image, clim=(0, 0.1), cmap='viridis', aspect='auto')
    plt.scatter(ys, xs, color='r')
    plt.plot(fit_total, xs, 'r')
    plt.legend()
    plt.show()


def filter_datapoints(data, dt):

    noisefilter = signal.gaussian(int(14/dt)+1, 2.5)
    noisefilter = noisefilter / np.sum(noisefilter)

    mean = data.mean()

    vals = np.zeros(len(data))
    ys = np.zeros(len(data))

    for i, row in enumerate(data):

        row_norm = row - mean
        # Set all values less than 5 std from mean to zero
        row_norm[row_norm < 6 * np.std(row_norm)] = 0
        # Apply gaussian filter
        row_norm = np.convolve(row_norm, noisefilter, mode='same')

        vals[i] = np.max(row_norm)
        ys[i] = np.argmax(row_norm)

    xs = np.arange(len(ys))

    return xs, ys, vals


def remove_outliers(xs, ys, vals, mean, band):
    ys_ = ys[(ys > mean - band) & (ys < mean + band)]
    xs_ = xs[(ys > mean - band) & (ys < mean + band)]
    vals_ = vals[(ys > mean - band) & (ys < mean + band)]
    return xs_, ys_, vals_

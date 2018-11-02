import sys
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


def process(dataid):
    xs, ys, vals, tca, fca = extract_datapoints(dataid)
    filepath = Config().paths['extracted'] / f"{dataid}.DOP1B"
    with open(filepath, 'w+') as file:
        file.write(f"tca: {tca}\n")
        file.write(f"fca: {fca}\n")
        file.write("==============\n")
        file.write('time,freq,vals\n')
        for x, y, val in zip(xs, ys, vals):
            file.write(f"{x},{y},{val}\n")


def extract_datapoints(dataid, plot=False):

    logger.info(f'Extracting frequency data points from {dataid}')
    spectrogram = Spectrogram.load(dataid)
    image = spectrogram.image

    # First masking
    mask1 = area_mask(image, ori='freq')
    mask2 = area_mask(image, ori='time')
    mask = np.logical_and(mask1, mask2)

    # Apply mask and change zeros to the mean value of the data
    image_masked = image * mask + np.mean(image) * np.logical_not(mask)

    # First fitting cycle
    sideband = 600
    widths = sideband * np.array([2.5, 1.5, 0.5])
    xs, ys, vals, fitfunc = fitting_cycle(image_masked,
                                          spectrogram.dt,
                                          widths=widths,
                                          cluster_eps=25,
                                          plot=plot)

    logger.info(f'1st fitting cycle completed.\n    Found {len(xs)} data points.')
    if plot:
        plt.figure()
        plt.imshow(image_masked, clim=(0, 0.1), cmap='viridis', aspect='auto')
        plt.scatter(ys, xs, color='r')
        times = np.arange(image.shape[0])
        plt.plot(fitfunc(times), times, 'r')
        plt.title('First masking and fitting cycle')
        plt.show()

    # Second masking
    mask1 = fit_mask(image, fitfunc)
    mask2 = time_mask(image, xs[0], xs[-1])
    mask = np.logical_and(mask1, mask2)
    image_masked = image_masked * mask

    # Second fitting cycle
    sideband = 600
    widths = sideband * np.array([0.5])
    xs, ys, vals, fitfunc = fitting_cycle(image_masked,
                                          spectrogram.dt,
                                          widths=widths,
                                          cluster_eps=15,
                                          tresh=0.05,
                                          plot=plot)

    logger.info(f'2nd fitting cycle completed.\n    Found {len(xs)} data points.')
    if plot:
        plt.figure()
        plt.imshow(image_masked, clim=(0, 0.1), cmap='viridis', aspect='auto')
        plt.scatter(ys, xs, color='r')
        times = np.arange(image.shape[0])
        plt.plot(fitfunc(times), times, 'r')
        plt.title('Second masking and fitting cycle')
        plt.show()

    # Calculate bounds for root finding
#    middle = xs[0] + (xs[-1] - xs[0]) / 2
#    bound = (xs[-1] - xs[0]) / 8
    # TODO Might fail if xs is sparse
    deriv2 = egrad(egrad(fitfunc))(xs)
    lower_bound = xs[deriv2.argmin()]
    upper_bound = xs[deriv2.argmax()]

    # Perform root finding to determine tca and fca
    if plot:
        plt.figure()
        x = np.linspace(xs[0], xs[-1], 1000)
        y = egrad(egrad(fitfunc))(x)
        plt.plot(x, y)
        plt.show()
#    try:
    tca = optimize.brentq(egrad(egrad(fitfunc)), lower_bound, upper_bound)
    fca = fitfunc(tca)
#    except:
#        logger.error('Could not find root to determine tca')
#        tca = None
#        fca = None

    logger.info(f'Additional values calculated:\n    tca: {tca}\n    fca: {fca}')

    plt.figure()
    plt.imshow(image, clim=(0, 0.1), cmap='viridis', aspect='auto')
    plt.scatter(ys, xs, color='r')
    plt.title(f'{dataid}: Final data points')
    plt.show()

    return xs, ys, vals, tca, fca


def fitting_cycle(image, dt, widths, cluster_eps, tresh=None, plot=False):

    xs, ys, vals = filter_datapoints_rowmaxes(image, dt)

    # Repeatedly fit tangent curve to filtered data points
    for width in widths:
        fit_coeffs = fit_tanh(xs, ys, dt)
        tanh_fit = tanh(xs, *fit_coeffs)
        xs, ys, vals = remove_width_outliers(xs, ys, vals, tanh_fit, width)

    if tresh:
        xs = xs[vals > tresh]
        ys = ys[vals > tresh]
        vals = vals[vals > tresh]

    tanh_fit = tanh(xs, *fit_coeffs)
    res = ys - tanh_fit

    # Find clusters of data points
    data = np.dstack((xs, res))[0]
    db = DBSCAN(eps=cluster_eps, min_samples=5).fit(data)
    data = np.column_stack(data)

    # Find labels of all clusters with 10 or more data points
    counter = Counter(db.labels_)
    counter.pop(-1, None)
    labels = []
    for key, count in counter.most_common():
        if count < 10:
            break
        else:
            labels.append(key)

    # Create temp datapoints and fit func only from clusters with 10 or more
    bools = np.isin(db.labels_, labels)
    xs_temp = data[0][bools]
    res_temp = data[1][bools]
    res_func_temp, res_coeffs_temp = fit_residual(xs_temp, res_temp)

    # Calculate residual between data and fit, and remove clusters far from fit
    res2 = ys - res_func_temp(xs, *res_coeffs_temp)
    final_labels = []
    for label in labels:
        if abs(np.mean(res2[db.labels_ == label])) > 15:
            final_labels.append(label)

    final_bools = np.isin(db.labels_, final_labels)
    xs = xs[final_bools]
    ys = ys[final_bools]
    vals = vals[final_bools]
    res = res[final_bools]
    res_func, res_coeffs = fit_residual(xs, res)

    def fit_func(x):
        """Calculates the fit for any value."""
        return tanh(x, *fit_coeffs) + res_func(x, *res_coeffs)

    if plot:
        plt.figure()
        for i in set(db.labels_):
            plt.scatter(data[0][db.labels_ == i], data[1][db.labels_ == i], label=i)
        times = np.linspace(xs[0], xs[-1], 1000)
        plt.plot(times, res_func(times, *res_coeffs))
        plt.legend()
        plt.show()

    return xs, ys, vals, fit_func


def filter_datapoints_rowmaxes(data, dt):

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


def remove_width_outliers(xs, ys, vals, mean, band):
    ys_ = ys[(ys > mean - band) & (ys < mean + band)]
    xs_ = xs[(ys > mean - band) & (ys < mean + band)]
    vals_ = vals[(ys > mean - band) & (ys < mean + band)]
    return xs_, ys_, vals_

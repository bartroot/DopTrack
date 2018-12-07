import matplotlib.pyplot as plt
import logging
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import scipy.signal as signal
import scipy.optimize as optimize
from scipy.stats import linregress
from collections import Counter
from sklearn.cluster import DBSCAN

from .masking import horizontal_mask, vertical_mask, spike_mask, fit_mask, time_mask
from .fitting import tanh, fit_tanh, fit_residual, FittingError


logger = logging.getLogger(__name__)


class PassNotFoundError(Exception):
    pass


def extract_frequency_data(spectrogram, dt, plot=False):

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f'Original spectrogram')
        clim = (0, spectrogram.mean() + 3.5*spectrogram.std())
        ax.imshow(spectrogram, clim=clim, cmap='viridis', aspect='auto')
        plt.show()

    # First extraction cycle
    masked_spectrogram = first_masking(spectrogram, dt, plot=plot)
    data = first_fitting_cycle(masked_spectrogram, dt, plot=plot)
    data = filter_data_points(data, plot=plot)
    logger.info(f"Extraction cycle 1 completed. Found {len(data['time'])} data points.")

    # Second extraction cycle
    masked_spectrogram = second_masking(masked_spectrogram, data['fit_func'], data['time'][0], data['time'][-1], dt, plot=plot)
    data = second_fitting_cycle(masked_spectrogram, dt, plot=plot)
    data = filter_data_points(data, plot=plot)
    logger.info(f"Extraction cycle 2 completed. Found {len(data['time'])} data points.")

    # Calculation of additional values
    data['tca'], data['fca'] = calc_tca_and_fca(data)
    data['rms'] = np.sqrt(((data['frequency'] - data['tanh_fit'])**2).mean())

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Final extracted data points')
        clim = (0, spectrogram.mean() + 3.5*spectrogram.std())
        ax.imshow(spectrogram, clim=clim, cmap='viridis', aspect='auto',
                  extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        ax.scatter(data['frequency'], data['time'], color='r')
        fig.show()

    return data


def create_fit_func(tanh_coeffs, residual_coeffs, residual_func):
        def fit_func(x):
            return tanh(x, *tanh_coeffs) + residual_func(x, *residual_coeffs)
        return fit_func


def create_tanh_func(tanh_coeffs):
        def tanh_func(x):
            return tanh(x, *tanh_coeffs)
        return tanh_func


def first_masking(spectrogram, dt, plot=False):
    mask1 = vertical_mask(spectrogram, dt)
    mask2 = horizontal_mask(spectrogram, dt)
    mask3 = spike_mask(spectrogram, dt)
    mask = np.logical_and(mask1, np.logical_and(mask2, mask3))
    masked_spectrogram = spectrogram * mask + np.mean(spectrogram) * np.logical_not(mask)

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Result of first masking')
        clim = (0, spectrogram.mean() + 3.5*spectrogram.std())
        ax.imshow(spectrogram * mask, clim=clim, cmap='viridis', aspect='auto',
                  extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        fig.show()
    return masked_spectrogram


def second_masking(spectrogram, fitfunc, start_of_signal, end_of_signal, dt, plot=False):
    mask1 = fit_mask(spectrogram, fitfunc, dt)
    mask2 = time_mask(spectrogram, start_of_signal, end_of_signal, dt)
    mask = np.logical_and(mask1, mask2)
    masked_spectrogram = spectrogram * mask + np.mean(spectrogram) * np.logical_not(mask)

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Result of second masking')
        clim = (0, spectrogram.mean() + 3.5*spectrogram.std())
        ax.imshow(spectrogram * mask, clim=clim, cmap='viridis', aspect='auto',
                  extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        fig.show()

    return masked_spectrogram


def first_fitting_cycle(spectrogram, dt, plot=False):

    # Extract initial data points
    data = get_rowmaxes_as_points(spectrogram, dt)

    # Repeatedly compute tanh fit and filter outliers
    sideband = 600
    widths = sideband * np.array([2.5, 1.5, 0.5])
    try:
        for width in widths:
            tanh_coeffs = fit_tanh(data['time'], data['frequency'], dt)
            tanh_fit = tanh(data['time'], *tanh_coeffs)
            data = remove_width_outliers(data, tanh_fit, width)
        data['tanh_coeffs'] = fit_tanh(data['time'], data['frequency'], dt)
        data['tanh_fit'] = tanh(data['time'], *data['tanh_coeffs'])
    except FittingError as e:
        raise PassNotFoundError(f'Fitting failed during first fitting cycle: {e}')

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Result of first fitting cycle')
        clim = (0, spectrogram.mean() + 3.5*spectrogram.std())
        ax.imshow(spectrogram, clim=clim, cmap='viridis', aspect='auto',
                  extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        ax.scatter(data['frequency'], data['time'], color='r')
        _x = np.linspace(data['time'][0], data['time'][-1], 1000)
        _y = tanh(_x, *data['tanh_coeffs'])
        ax.plot(_y, _x, label='tanh fit', color='k')
        ax.set_ylim(int(spectrogram.shape[0]*dt), 0)
        ax.legend()
        fig.show()

    return data


def second_fitting_cycle(spectrogram, dt, plot=False):

    # Extract initial data points and remove points with too low power
    data = get_rowmaxes_as_points(spectrogram, dt)
    treshold = spectrogram.mean() + 2*spectrogram.std()
    data['time'] = data['time'][data['power'] > treshold]
    data['frequency'] = data['frequency'][data['power'] > treshold]
    data['power'] = data['power'][data['power'] > treshold]
    if len(data['time']) < 10:
        raise PassNotFoundError('No or few data points left after filtering by power level')

    # Compute tanh fit
    try:
        data['tanh_coeffs'] = fit_tanh(data['time'], data['frequency'], dt)
        data['tanh_fit'] = tanh(data['time'], *data['tanh_coeffs'])
    except FittingError as e:
        raise PassNotFoundError(f'Fitting failed during second fitting cycle: {e}')

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Result of second fitting cycle')
        clim = (0, spectrogram.mean() + 3.5*spectrogram.std())
        ax.imshow(spectrogram, clim=clim, cmap='viridis', aspect='auto',
                  extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        ax.scatter(data['frequency'], data['time'], color='r')
        _x = np.linspace(data['time'][0], data['time'][-1], 1000)
        _y = tanh(_x, *data['tanh_coeffs'])
        ax.plot(_y, _x, label='tanh fit', color='k')
        ax.set_ylim(int(spectrogram.shape[0]*dt), 0)
        ax.legend()
        fig.show()

    return data


def filter_data_points(data, plot=False):

    # Remove clusters with two few points or with vertical lines of points
    data['residual'] = data['frequency'] - data['tanh_fit']
    data['labels'] = create_clusters(data['time'], data['residual'], eps=25, min_samples=5, plot=plot)
    labels1 = labels_of_clusters_with_n_or_more_points(data['labels'], n=10)
    labels2 = labels_of_clusters_with_nonzero_slope(data['labels'], data['time'], data['frequency'])
    data = filter_clusters_by_label(data, set.intersection(labels1, labels2))

    # Calculate initial residual fit
    data['residual'] = data['frequency'] - data['tanh_fit']
    try:
        data['residual_func'], data['residual_coeffs'] = fit_residual(data['time'], data['residual'])
    except FittingError as e:
        raise PassNotFoundError(f'Fitting failed during filtering: {e}')
    # Remove clusters where mean of second residual is too high
    data['labels'] = create_clusters(data['time'], data['residual'], eps=25, min_samples=5, plot=plot)
    second_residual = data['residual'] - data['residual_func'](data['time'], *data['residual_coeffs'])
    labels3 = labels_of_clusters_with_low_mean(second_residual, data['labels'], mean=15)
    data = filter_clusters_by_label(data, labels3)

    # Calculate final residual fit and add fits to final data
    try:
        data['residual_func'], data['residual_coeffs'] = fit_residual(data['time'], data['residual'])
    except FittingError as e:
        raise PassNotFoundError(f'Fitting failed during filtering: {e}')
    data['fit_func'] = create_fit_func(data['tanh_coeffs'], data['residual_coeffs'], data['residual_func'])
    data['final_fit'] = data['fit_func'](data['time'])

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Filtered data points and function fits')
        ax.scatter(data['frequency'], data['time'], color='r')
        ax.plot(data['tanh_fit'], data['time'], label='tanh fit')
        ax.plot(data['final_fit'], data['time'], label='tanh+residual fit')
        ax.invert_yaxis()
        ax.legend()
        fig.show()

    return data


def create_clusters(xs, ys, eps, min_samples, plot=False):
    data = np.dstack((xs, ys))[0]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = clustering.labels_
    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f'Clustering with eps={eps} and min_samples={min_samples}')
        for label in set(labels):
            plt.plot(xs[labels == label], ys[labels == label], '.', label=label)
        ax.legend()
        fig.show
    logger.debug(f'Found {len(set(labels))} clusters: {set(labels)}')
    return labels


def calc_tca_and_fca(data):

    deriv2 = egrad(egrad(data['fit_func']))(data['time'])
    lower_bound = data['time'][deriv2.argmin()]
    upper_bound = data['time'][deriv2.argmax()]

    try:
        tca = optimize.brentq(egrad(egrad(data['fit_func'])), lower_bound, upper_bound)
    except ValueError:
        logger.warning("Root finding failed with fit_func, using tanh instead.")
        # TODO maybe refactor this so tanh_func is in data dict
        tanh_func = create_tanh_func(data['tanh_coeffs'])
        # Also expanding bounds
        tca = optimize.brentq(egrad(egrad(tanh_func)), lower_bound - 1000, upper_bound + 1000)
    fca = data['fit_func'](tca)

    logger.debug(f"Values at closest approach calculated. tca:{round(tca, 2)} fca:{round(fca, 2)}")
    return tca, fca


def labels_of_clusters_with_n_or_more_points(array_of_labels, n):
    counter = Counter(array_of_labels)
    counter.pop(-1, None)
    valid_labels = set()
    for key, count in counter.most_common():
        if count < n:
            break
        else:
            valid_labels.add(key)
    logger.debug(f'Found {len(valid_labels)} clusters with {n}+ data points: {valid_labels}')
    return valid_labels


def labels_of_clusters_with_low_mean(values, array_of_labels, mean):
    valid_labels = set()
    for label in set(array_of_labels):
        if abs(np.mean(values[array_of_labels == label])) < mean:
            valid_labels.add(label)
    logger.debug(f'Found {len(valid_labels)} clusters with mean of 2nd residual lower than {mean}: {valid_labels}')
    return valid_labels


def labels_of_clusters_with_nonzero_slope(array_of_labels, xs, ys):
    """"Added to counter cases where vertical masking is not effective enough."""
    valid_labels = set()
    for label in set(array_of_labels):
        if label != -1:
            linreg = linregress(xs[array_of_labels == label], ys[array_of_labels == label])
            if abs(linreg.slope) > 1:
                valid_labels.add(label)
    logger.debug(f'Found {len(valid_labels)} clusters with non-zero slope: {valid_labels}')
    return valid_labels


def filter_clusters_by_label(data, labels_to_filter):
    bools = np.isin(data['labels'], list(labels_to_filter))
    if len(bools[bools == True]) == 0:
        raise PassNotFoundError('No data points found after filtering clusters')
    keys = [key for key in data.keys() if type(data[key]) == np.ndarray]
    for key in keys:
        if len(data[key]) == len(bools):
            data[key] = data[key][bools]
    return data


def get_rowmaxes_as_points(spectrogram, dt):

    noisefilter = signal.gaussian(int(14 / dt) + 1, 2.5)
    noisefilter = noisefilter / np.sum(noisefilter)

    mean = spectrogram.mean()

    power = np.zeros(len(spectrogram))
    frequency = np.zeros(len(spectrogram))

    for i, row in enumerate(spectrogram):
        row_norm = row - mean
        # Set all values less than 5 std from mean to zero
        row_norm[row_norm < 5 * np.std(row_norm)] = 0
        # Apply gaussian filter
        row_norm = np.convolve(row_norm, noisefilter, mode='same')

        power[i] = np.max(row_norm)
        frequency[i] = np.argmax(row_norm)

    time = np.arange(len(frequency)) * dt + 0.5 * dt

    # Filter out data points where freq is zero
    bools = frequency != 0
    time = time[bools]
    frequency = frequency[bools]
    power = power[bools]

    return {'time': time, 'frequency': frequency, 'power': power}


def remove_width_outliers(data, mean, band):
    bools = (data['frequency'] > mean - band) & (data['frequency'] < mean + band)
    new_frequency = data['frequency'][bools]
    new_time = data['time'][bools]
    new_power = data['power'][bools]
    return {'time': new_time, 'frequency': new_frequency, 'power': new_power}

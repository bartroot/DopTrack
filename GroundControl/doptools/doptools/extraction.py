import logging
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import scipy.signal as signal
import scipy.optimize as optimize
from collections import Counter
from sklearn.cluster import DBSCAN

from .masking import area_mask, fit_mask, time_mask
from .fitting import tanh, fit_tanh, fit_residual


logger = logging.getLogger(__name__)


def extract_frequency_data(image, dt):

    # First extraction cycle
    image = first_masking(image)
    data = first_fitting_cycle(image, dt)
    fit_func = create_fit_func(data['tanh_coeffs'],
                               data['residual_coeffs'],
                               data['residual_func'])
    logger.debug(f"Extraction cycle 1 completed. Found {len(data['time'])} data points.")

    # Second extraction cycle
    image = second_masking(image, fit_func, data['time'][0], data['time'][-1], dt)
    data = second_fitting_cycle(image, dt)
    fit_func = create_fit_func(data['tanh_coeffs'],
                               data['residual_coeffs'],
                               data['residual_func'])
    data['fit_func'] = fit_func
    logger.debug(f"Extraction cycle 2 completed. Found {len(data['time'])} data points.")

    # Calculation of additional values
    data['tca'] = calc_tca(data['time'], data['fit_func'], data['tanh_coeffs'])
    data['fca'] = fit_func(data['tca'])
    logger.debug(f"Values at closest approach calculated. tca:{data['tca']} fca:{data['fca']}")

    return data


def create_fit_func(tanh_coeffs, residual_coeffs, residual_func):
        def fit_func(x):
            return tanh(x, *tanh_coeffs) + residual_func(x, *residual_coeffs)
        return fit_func


def create_tanh_func(tanh_coeffs):
        def tanh_func(x):
            return tanh(x, *tanh_coeffs)
        return tanh_func


def first_masking(image):
    mask1 = area_mask(image, ori='freq')
    mask2 = area_mask(image, ori='time')
    mask = np.logical_and(mask1, mask2)
    masked_image = image * mask + np.mean(image) * np.logical_not(mask)
    return masked_image


def second_masking(image, fitfunc, start_of_signal, end_of_signal, dt):
    mask1 = fit_mask(image, fitfunc, dt)
    mask2 = time_mask(image, start_of_signal, end_of_signal, dt)
    mask = np.logical_and(mask1, mask2)
    masked_image = image * mask
    return masked_image


def first_fitting_cycle(image, dt):

    # Extract initial data points
    time, frequency, power = get_rowmaxes_as_points(image, dt)

    # Repeatedly fit tangent curve and filter data points
    sideband = 600
    widths = sideband * np.array([2.5, 1.5, 0.5])

    for width in widths:
        tanh_coeffs = fit_tanh(time, frequency, dt)
        tanh_fit = tanh(time, *tanh_coeffs)
        time, frequency, power = remove_width_outliers(time, frequency, power, tanh_fit, width)
    tanh_fit = tanh(time, *tanh_coeffs)
    residual = frequency - tanh_fit

    # Find clusters of data points
    data = np.dstack((time, residual))[0]
    clustering = DBSCAN(eps=25, min_samples=5).fit(data)
    data = {'time': time,
            'frequency': frequency,
            'power': power,
            'residual': residual,
            'labels': clustering.labels_}

    # Remove clusters with too few points
    filtered_labels = labels_of_clusters_with_n_or_more_points(data['labels'], n=10)
    data = filter_clusters_by_label(data, filtered_labels)

    # Calculate initial residual fit
    residual_func, residual_coeffs = fit_residual(data['time'], data['residual'])

    # Remove cluster where mean of residual of residual is too high
    residual_of_residual = data['residual'] - residual_func(data['time'], *residual_coeffs)
    final_labels = labels_of_clusters_with_low_mean(residual_of_residual, data['labels'], mean=15)
    data = filter_clusters_by_label(data, final_labels)

    # Calculate final residual fit and add fits to final data
    residual_func, residual_coeffs = fit_residual(time, residual)
    data['tanh_coeffs'] = tanh_coeffs
    data['residual_func'] = residual_func
    data['residual_coeffs'] = residual_coeffs

    return data


def second_fitting_cycle(image, dt):

    # Extract initial data points and remove points with too low power
    time, frequency, power = get_rowmaxes_as_points(image, dt)
    treshold = 0.05
    time = time[power > treshold]
    frequency = frequency[power > treshold]
    power = power[power > treshold]

    # Compute initial fit
    tanh_coeffs = fit_tanh(time, frequency, dt)
    tanh_fit = tanh(time, *tanh_coeffs)
    residual = frequency - tanh_fit

    # Find clusters of data points
    data = np.dstack((time, residual))[0]
    clustering = DBSCAN(eps=25, min_samples=5).fit(data)
    data = {'time': time,
            'frequency': frequency,
            'power': power,
            'residual': residual,
            'labels': clustering.labels_}

    # Remove clusters with too few points
    filtered_labels = labels_of_clusters_with_n_or_more_points(data['labels'], n=10)
    data = filter_clusters_by_label(data, filtered_labels)

    # Calculate initial residual fit
    residual_func, residual_coeffs = fit_residual(data['time'], data['residual'])

    # Remove cluster where mean of residual of residual is too high
    residual_of_residual = data['residual'] - residual_func(data['time'], *residual_coeffs)
    final_labels = labels_of_clusters_with_low_mean(residual_of_residual, data['labels'], mean=15)
    data = filter_clusters_by_label(data, final_labels)

    # Calculate final residual fit and add fits to final data
    residual_func, residual_coeffs = fit_residual(time, residual)
    data['tanh_coeffs'] = tanh_coeffs
    data['residual_func'] = residual_func
    data['residual_coeffs'] = residual_coeffs

    return data


def calc_tca(time, fit_func, tanh_coeffs):

    deriv2 = egrad(egrad(fit_func))(time)
    lower_bound = time[deriv2.argmin()]
    upper_bound = time[deriv2.argmax()]

    try:
        tca = optimize.brentq(egrad(egrad(fit_func)), lower_bound, upper_bound)
    except ValueError:
        logger.warning("Root finding failed with fit_func, using tanh instead.")
        # TODO maybe refactor this so tanh_func is in data dict
        tanh_func = create_tanh_func(tanh_coeffs)
        # Also expanding bounds
        tca = optimize.brentq(egrad(egrad(tanh_func)), lower_bound - 1000, upper_bound + 1000)

    return tca


def labels_of_clusters_with_n_or_more_points(array_of_labels, n):
    counter = Counter(array_of_labels)
    counter.pop(-1, None)
    new_labels = []
    for key, count in counter.most_common():
        if count < n:
            break
        else:
            new_labels.append(key)
    return new_labels


def labels_of_clusters_with_low_mean(values, array_of_labels, mean):
    new_labels = []
    for label in set(array_of_labels):
        if abs(np.mean(values[array_of_labels == label])) < mean:
            new_labels.append(label)
    return new_labels


def filter_clusters_by_label(data, labels_to_filter):
    bools = np.isin(data['labels'], labels_to_filter)
    for key in data.keys():
        data[key] = data[key][bools]
    return data


def get_rowmaxes_as_points(image, dt):

    noisefilter = signal.gaussian(int(14 / dt) + 1, 2.5)
    noisefilter = noisefilter / np.sum(noisefilter)

    mean = image.mean()

    power = np.zeros(len(image))
    frequency = np.zeros(len(image))

    for i, row in enumerate(image):
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

    return time, frequency, power


def remove_width_outliers(time, frequency, power, mean, band):
    bools = (frequency > mean - band) & (frequency < mean + band)
    new_frequency = frequency[bools]
    new_time = time[bools]
    new_power = power[bools]
    return new_time, new_frequency, new_power

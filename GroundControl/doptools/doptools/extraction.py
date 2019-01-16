"""Frequency data extraction

This module contains functions for extracting time-frequency data points of a
satellite pass from a spectrograms.

Classes
-------
- `PassNotFoundError` -- Exception thrown when no pass is found.

Routines
--------
- `extract_frequency_data` -- Main extraction function.
- `create_fit_func` -- Create a fitting function from Tanh and residual fit.
- `first_masking` -- Create specgtrogram mask for the initial extraction cycle.
- `second_masking` -- Create specgtrogram mask for the extraction refinement cycle.
- `create_clusters` -- Create clusters of data points.
- `estimate_tca` -- Estimate time of closest approach of the satellite pass.
- `labels_of_clusters_with_n_or_more_points`
- `labels_of_clusters_with_low_mean`
- `labels_of_clusters_with_negative_slope`
- `filter_clusters_by_label`
- `get_rowmaxes_as_points` -- Extract the maximum pixel of each row as data points.
- `remove_width_outliers` -- Remove data points outside some band in spectrogram.

Warnings
-----
The current implementation of the extraction algorithm is designed specifically
for recordings of the Delfi-C3 satellie signal. Other satellite signals with
different modulation technique will not be extracted correctly.

"""
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
    """Raised whenever the extraction algorithm decides no pass is found in the spectrogram."""
    pass


def extract_frequency_data(spectrogram, dt, plot=False):
    """
    Extract frequency data from a spectrogram.

    Parameters
    ----------
    spectrogram : (N, M) numpy.ndarray
        A 2D array with the spectrogram data.
    dt : int or float
        The timestep of the spectrogram.
    plot : bool, optional
        If True plot a figure of the resulting clusters.

    Returns
    -------
    dict
        Resulting data including frequency series, fitting coefficients, etc.

    Raises
    ------
    PassNotFoundError
        If extraction algorithm decides no pass in present in the recorded spectrogram.
        Raised under multiple different situations.

    Notes
    -----
    The extraction function can extract frequency data from spectrograms with
    different timesteps, but the current algorithm is specifically designed
    to work best with a timestep of 0.1 second. The script will run succesfully
    with other timesteps, but the results might be suboptimal or even bad.

    While timesteps close to 0.1, e.g. 0.2 or 0.3, should work fine, a more
    robust algorithm should be developed if good results are needed for
    timesteps in the full desired range from ~2 seconds to 0.1 seconds.
    """

    original_spectrogram = spectrogram  # The original is only kept for plotting purposes
    spectrogram = np.copy(spectrogram)

    #####################################################################################
    ############################# INITIAL EXTRACTION CYCLE ##############################
    #####################################################################################

    spectrogram = first_masking(spectrogram, dt, plot=plot)
    data = get_rowmaxes_as_points(spectrogram, dt, cutoff=0.02, plot=plot)

    # Remove clusters with two few points or with vertical lines of points
    data['labels'] = create_clusters(
            data['time'],
            data['frequency']/14,  # Scaled down so both axes have similar scale
            eps=30*np.sqrt(dt),  # 20 is too low, 30 might be too high
            min_samples=5,
            plot=plot
            )
    labels = labels_of_clusters_with_negative_slope(data['labels'], data['time'], data['frequency'])
    data = filter_clusters_by_label(data, labels, exclude_outliers=True, plot=plot)

    # Repeatedly compute tanh fit and filter outliers
    sideband = 600
    widths = sideband * np.array([5, 3, 1])
    try:
        for width in widths:
            tanh_coeffs = fit_tanh(data['time'], data['frequency'], dt)
            tanh_fit = tanh(data['time'], *tanh_coeffs)
            data = remove_width_outliers(data, tanh_fit, width)
        tanh_coeffs = fit_tanh(data['time'], data['frequency'], dt)
    except FittingError as e:
        raise PassNotFoundError(f'Fitting failed during first fitting cycle: {e}')

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Data points and fit after rough extraction')
        clim = (0, original_spectrogram.mean() + 3.5*original_spectrogram.std())
        ax.imshow(
                original_spectrogram,
                clim=clim,
                cmap='viridis',
                aspect='auto',
                extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        ax.scatter(data['frequency'], data['time'], color='r')
        ax.plot(tanh(data['time'], *tanh_coeffs), data['time'], color='g')
        fig.show()

    data['residual'] = data['frequency'] - tanh(data['time'], *tanh_coeffs)
    data['labels'] = create_clusters(
            data['time'],
            data['residual'],
            eps=50*np.sqrt(dt),
            min_samples=5,
            plot=plot
            )
    labels1 = labels_of_clusters_with_n_or_more_points(data['labels'], n=int(15/np.sqrt(dt)))
    labels2 = labels_of_clusters_with_negative_slope(data['labels'], data['time'], data['frequency'])
    data = filter_clusters_by_label(data, set.intersection(labels1, labels2), exclude_outliers=True)

    # Calculate initial residual fit
    try:
        residual_func, residual_coeffs = fit_residual(data['time'], data['residual'])
    except FittingError as e:
        raise PassNotFoundError(f'Fitting failed during filtering: {e}')
    fit_func = create_fit_func(tanh_coeffs, residual_func, residual_coeffs)

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f'Residual fit')
        ax.scatter(data['time'], data['residual'], color='r')
        ax.plot(data['time'], residual_func(data['time'], *residual_coeffs))
        fig.show()

    #####################################################################################
    ########################### EXTRACTION REFINEMENT CYCLE #############################
    #####################################################################################

    spectrogram = second_masking(spectrogram, fit_func, data['time'][0], data['time'][-1], dt, plot=plot)
    data = get_rowmaxes_as_points(spectrogram, dt, cutoff=0.005, plot=plot)

    data['residual'] = data['frequency'] - tanh(data['time'], *tanh_coeffs)
    data['second_residual'] = data['residual'] - residual_func(data['time'], *residual_coeffs)
    data['labels'] = create_clusters(
            data['time'], data['second_residual'],
            eps=20*np.sqrt(dt),
            min_samples=5,
            plot=plot)

    labels1 = labels_of_clusters_with_low_mean(data['labels'], data['second_residual'], maximum_mean=50)
    labels2 = labels_of_clusters_with_n_or_more_points(data['labels'], n=20)
    data = filter_clusters_by_label(data, set.intersection(labels1, labels2), exclude_outliers=True)

    # Final check to see if there is enough data points to be an actual pass
    if len(data['time']) < 100/np.sqrt(dt):
        raise PassNotFoundError(f'Final extraction contains too few data points: {len(data["time"])}<{100/np.sqrt(dt)}')

    # Calculate final fits
    tanh_coeffs = fit_tanh(data['time'], data['frequency'], dt)
    residual_func, residual_coeffs = fit_residual(data['time'], data['residual'])
    fit_func = create_fit_func(tanh_coeffs, residual_func, residual_coeffs)

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Filtered data points and function fits')
        ax.scatter(data['frequency'], data['time'], color='r')
        ax.plot(tanh(data['time'], *tanh_coeffs), data['time'], label='tanh fit')
        ax.plot(fit_func(data['time']), data['time'], label='tanh+residual fit')
        ax.invert_yaxis()
        ax.legend()
        fig.show()

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Final extracted data points')
        clim = (0, original_spectrogram.mean() + 3.5*original_spectrogram.std())
        ax.imshow(
                original_spectrogram,
                clim=clim,
                cmap='viridis',
                aspect='auto',
                extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        ax.scatter(data['frequency'], data['time'], color='r')
        t = np.linspace(0, 700, 500)
        ax.plot(tanh(t, *tanh_coeffs), t, 'g')
        fig.show()

    data['tca'] = estimate_tca(data, fit_func, tanh_coeffs)
    data['fca'] = fit_func(data['tca'])
    data['rmse'] = np.sqrt(((data['frequency'] - tanh(data['time'], *tanh_coeffs))**2).mean())
    data['tanh_coeffs'] = tanh_coeffs
    data['residual_func'] = residual_func
    data['residual_coeffs'] = residual_coeffs

    return data


def create_fit_func(tanh_coeffs, base_residual_func, residual_coeffs):
    """
    Create final fitting function.

    Parameters
    ----------
    tanh_coeffs : list
        Coefficients defining the tanh function.
    base_residual_func : func
        Base function used during residual fitting.
        Needs coefficients as input to be the complete residual function.
    residual_coeffs : list
        Coefficients defining the specific residual function.

    Returns
    -------
    func
        The resulting fitting function.
    """
    def fit_func(x):
        return tanh(x, *tanh_coeffs) + base_residual_func(x, *residual_coeffs)
    return fit_func


def first_masking(spectrogram, dt, plot=False):
    """
    Perform first masking of spectrogram.

    Parameters
    ----------
    spectrogram : (N, M) numpy.ndarray
        A 2D array of the spectrogram.
    dt : int or float
        The timestep of the spectrogram.
    plot : bool, optional
        If True plot a figure of the masked spectrogram.

    Returns
    -------
    (N, M) numpy.ndarray
        The masked spectrogram.
    """
    logger.debug('Creating vertical mask...')
    mask1 = vertical_mask(spectrogram, dt)
    logger.debug('Creating horizontal mask...')
    mask2 = horizontal_mask(spectrogram, dt)
    logger.debug('Creating spike mask...')
    mask3 = spike_mask(spectrogram, dt)
    logger.debug('Applying first group of masks')
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


def second_masking(spectrogram, fit_func, start_of_signal, end_of_signal, dt, plot=False):
    """
    Perform second masking of spectrogram.

    Parameters
    ----------
    spectrogram : (N, M) numpy.ndarray
        A 2D array of the spectrogram.
    fit_func : func
        Function fitted to the data.
    start_of_signal: int or float
        Beginning of the signal in seconds from beginning of recording.
    end_of_signal : int or float
        End of signal in seconds from beginning of recording
    dt : int or float
        The timestep of the spectrogram.
    plot : bool, optional
        If True plot a figure of the masked spectrogram.

    Returns
    -------
    (N, M) numpy.ndarray
        The masked spectrogram.
    """
    logger.debug('Creating fit mask...')
    mask1 = fit_mask(spectrogram, fit_func, dt, bandwidth=1000)
    logger.debug('Creating time mask...')
    mask2 = time_mask(spectrogram, start_of_signal, end_of_signal, dt)
    logger.debug('Applying second group of masks...')
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


def create_clusters(xs, ys, eps, min_samples, plot=False):
    """
    Determine clusters in the data using the DBSCAN clustering algorithm.

    Parameters
    ----------
    xs : (N,) numpy.ndarray
        The x values of the data.
    ys : (N,) numpy.ndarray
        The y values of the data.
    eps : int or float
        Value of epsilon used in the DBSCAN algorithm.
   min_samples : int
        The number of samples defining a core point used in the DBSCAN algorithm.
    plot : bool, optional
        If True plot a figure of the resulting clusters.

    Returns
    -------
    (N,) numpy.ndarray
        A 1D array of the cluster label of each data point.
    """
    data = np.dstack((xs, ys))[0]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    # TODO the OPTICS algorithm might be more robust. Will release in sklearn 0.21.
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


def estimate_tca(data, fit_func, tanh_coeffs):
    """
    Estimates time of closest approach of the satellite pass from the frequency data points.

    Parameters
    ----------
    data : dict
        The x values of the data.
    fit_func : func
        Function fitted to the data.
    tanh_coeffs : list
        Coefficients of the fitted tanh function.

    Returns
    -------
    float
        Time of closest approach in seconds relative to beginning of recording.

    Raises
    ------
    PassNotFoundError
        If time of closest approach is outside of the time range of data points.
    """
    lower_bound = -1e4
    upper_bound = 1e4

    def tanh_func(x):
        return tanh(x, *tanh_coeffs)
    tca = optimize.brentq(egrad(egrad(tanh_func)), lower_bound, upper_bound)
    try:
        lower_bound = tca - 20
        upper_bound = tca + 20
        tca = optimize.brentq(egrad(egrad(fit_func)), lower_bound, upper_bound)
    except ValueError:
        logger.warning("Refined root finding failed with fit_func. Keeping result from tanh instead.")

    logger.debug(f"Time of closest approach estimated: {round(tca, 2)}")
    if not data['time'][0] < tca < data['time'][-1]:
        raise PassNotFoundError("TCA is outside the time range of the data points. TCA will be incorrect and pass is discarded.")

    return tca


def labels_of_clusters_with_n_or_more_points(array_of_labels, n):
    """
    Find labels of clusters with n or more points.

    Parameters
    ----------
    array_of_labels : (N,) numpy.ndarray
        Array of the cluster label of each data point.
    n : int
        Minimum number of data points for cluster to be valid.

    Returns
    -------
    set
        Labels of clusters with equal or more than n data points.
    """
    counter = Counter(array_of_labels)
    valid_labels = set()
    for key, count in counter.most_common():
        if count < n:
            break
        else:
            valid_labels.add(key)
    logger.debug(f'Found {len(valid_labels)} clusters with {n}+ data points: {valid_labels}')
    return valid_labels


def labels_of_clusters_with_low_mean(array_of_labels, values, maximum_mean):
    """
    Find labels of clusters with mean of some paramater lower than a maximum mean.

    Parameters
    ----------
    array_of_labels : (N,) numpy.ndarray
        Array of the cluster label of each data point.
    values : (N,) numpy.ndarray
        Values which should be compared to the mean the values.
    mean : float
        Maximum mean of data points for cluster to be valid.

    Returns
    -------
    set
        Labels of clusters with mean lower than maximum mean.
    """
    valid_labels = set()
    for label in set(array_of_labels):
        if abs(np.mean(values[array_of_labels == label])) < maximum_mean:
            valid_labels.add(label)
    logger.debug(f'Found {len(valid_labels)} clusters with mean lower than {maximum_mean}: {valid_labels}')
    return valid_labels


def labels_of_clusters_with_negative_slope(array_of_labels, xs, ys, min_slope=0.5):
    """
    Find labels of clusters with negative slope.

    Slope is found using linear regression. A absolute minimum slope value is also
    specified to remove clusters which are vertical, i.e. have constant frequency.

    Parameters
    ----------
    array_of_labels : (N,) numpy.ndarray
        Array of the cluster label of each data point.
    xs : (N,) numpy.ndarray
        The x values of the data.
    ys : (N,) numpy.ndarray
        The y values of the data.
    min_slope : float, optional
        The minimum aboslute value of the slope. A value of zero means that all clusters
        with negative slope are kept.

    Returns
    -------
    set
        Labels of clusters with mean lower than maximum mean.
    """
    valid_labels = set()
    for label in set(array_of_labels):
        linreg = linregress(xs[array_of_labels == label], ys[array_of_labels == label])
        if abs(linreg.slope) < min_slope:
            valid_labels.add(label)
    logger.debug(f'Found {len(valid_labels)} clusters with slope smaller than {- min_slope}: {valid_labels}')
    return valid_labels


def filter_clusters_by_label(data, labels_to_filter, exclude_outliers=True, plot=False):
    """
    Filter data by cluster labels to remove non-valid clusters.

    Parameters
    ----------
    data : dict
        Data of the extraction process.
    labels_to_filter : array_like
        Labels of all valid clusters, i.e. clusters to keep.
    exclude_outliers : bool, optional
        If True data points with label (-1) are removed. The label -1 is given to
        outliers during clustering.
    plot : bool, optional
        If True plot a figure of clusters remaining after filtering.

    Returns
    -------
    set
        Labels of clusters with mean lower than maximum mean.
    """
    if exclude_outliers:
        labels_to_filter.discard(-1)
    logger.debug(f'Filtering {len(labels_to_filter)} clusters: {labels_to_filter}')
    bools = np.isin(data['labels'], list(labels_to_filter))
    if len(bools[bools == True]) == 0:
        raise PassNotFoundError('No data points found after filtering clusters')
    keys = [key for key in data.keys() if type(data[key]) == np.ndarray]
    for key in keys:
        if len(data[key]) == len(bools):
            data[key] = data[key][bools]

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f'Data points after filtering clusters')
        for label in set(data['labels']):
            plt.plot(
                    data['time'][data['labels'] == label],
                    data['frequency'][data['labels'] == label],
                    '.',
                    label=label
                    )
        ax.legend()
        fig.show

    return data


def get_rowmaxes_as_points(spectrogram, dt, cutoff, plot=False):
    """
    Get the pixel with maximum power of each row in spectrogram as a data point.

    Parameters
    ----------
    spectrogram : (N, M) numpy.ndarray
        A 2D array of the spectrogram.
    dt : int or float
        The timestep of the spectrogram.
    cutoff : float
        Any pixel in the spectrogram lower than this cutoff value will be set to zero.
    plot : bool, optional
        If True plot a figure of resulting data points.

    Returns
    -------
    dict
        Dictionary with resulting data.
    """
    noisefilter = signal.gaussian(int(14 / dt) + 1, 10)
    noisefilter = noisefilter / np.sum(noisefilter)

    power = np.zeros(len(spectrogram))
    frequency = np.zeros(len(spectrogram))

    for i, row_original in enumerate(spectrogram):
        row = np.copy(row_original)
        row[row < cutoff] = 0

        # Apply gaussian filter
        row = np.convolve(row, noisefilter, mode='same')

        power[i] = np.max(row)
        frequency[i] = np.argmax(row)

    time = np.arange(len(frequency)) * dt + 0.5 * dt

    # Filter out data points where freq is zero
    bools = frequency > 100  # "Hacky" way to exclude points from input rows with constant values
    time = time[bools]
    frequency = frequency[bools]
    power = power[bools]

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Extracted maxima of rows')
        clim = (0, spectrogram.mean() + 3.5*spectrogram.std())
        ax.imshow(
                spectrogram,
                clim=clim,
                cmap='viridis',
                aspect='auto',
                extent=(0, spectrogram.shape[1], spectrogram.shape[0]*dt, 0))
        ax.scatter(frequency, time, color='r')
        fig.show()

    return {'time': time, 'frequency': frequency, 'power': power}


def remove_width_outliers(data, mean, bandwidth):
    """
    Remove data points that are outside a band defined by a mean value and a bandwidth.

    The bandwidth is fixed, but the mean is given as an array which allows the band to be curved.

    Parameters
    ----------
    data : dict
        Data of the extraction process.
    mean : (N,) numpy.ndarray
        An array of the mean values of the band.
    bandwidth : float
        The bandwidth of the band.

    Returns
    -------
    dict
        Dictionary with resulting data.
    """
    bools = (data['frequency'] > mean - bandwidth/2) & (data['frequency'] < mean + bandwidth/2)

    new_frequency = data['frequency'][bools]
    new_time = data['time'][bools]
    new_power = data['power'][bools]

    return {'time': new_time, 'frequency': new_frequency, 'power': new_power}

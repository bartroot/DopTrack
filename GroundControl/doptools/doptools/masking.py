"""Masking functions for time-frequncy data extraction.

Routines
--------
- `vertical mask`
- `horizontal_mask`
- `spike_mask`
- `fit_mask`
- `time_mask`

"""
import numpy as np
import cv2


def vertical_mask(image, dt):
    """
    Create a mask that removes vertical lines from a spectrogram.

    Parameters
    ----------
    image : (N, M) numpy.ndarray
        The image/spectrogram to be masked.
    dt : int or float
        The timestep of the spectrogram.
        Only used to give an initial guess of the coefficients.

    Returns
    -------
    (N, M) numpy.ndarray
        The mask.
    """
    coeffs = {'thresh': 0,
              'erode': (int(np.ceil(50/np.sqrt(dt))), 1),
              'dilate': (int(np.ceil(100/dt)), 15)}

    # TODO should be changed from depending on std to some fixed value
    mask = image > np.mean(image) + coeffs['thresh'] * np.std(image)

    # Erode the mask to find the location of areas with highest values
    kernel = np.ones(coeffs['erode'], np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel)

    # Dilate the mask to find approximate areas with highest values
    kernel = np.ones(coeffs['dilate'], np.uint8)
    mask = cv2.dilate(mask, kernel)

    # Invert ones and zeros to get the correct mask
    mask = np.logical_not(mask)

    return mask


def horizontal_mask(image, dt):
    """
    Create a mask that removes horizontal lines from a spectrogram.

    Parameters
    ----------
    image : (N, M) numpy.ndarray
        The image/spectrogram to be masked.
    dt : int or float
        The timestep of the spectrogram.
        Only used to give an initial guess of the coefficients.

    Returns
    -------
    (N, M) numpy.ndarray
        The mask.
    """
    coeffs = {'thresh': 5.0,
              'erode': (int(np.ceil(1)), 100),
              'dilate': (int(np.ceil(3/dt)), 1000)}

    # TODO should be changed from depending on std to some fixed value
    mask = image > np.mean(image) + coeffs['thresh'] * np.std(image)

    # Erode the mask to find the location of areas with highest values
    kernel = np.ones(coeffs['erode'], np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel)

    # Dilate the mask to find approximate areas with highest values
    kernel = np.ones(coeffs['dilate'], np.uint8)
    mask = cv2.dilate(mask, kernel)

    # Invert ones and zeros to get the correct mask
    mask = np.logical_not(mask)

    return mask


def spike_mask(image, dt):
    """
    Create a mask that removes areas around spikes in power level in a spectrogram.

    Parameters
    ----------
    image : (N, M) numpy.ndarray
        The image/spectrogram to be masked.
    dt : int or float
        The timestep of the spectrogram.
        Only used to give an initial guess of the coefficients.

    Returns
    -------
    (N, M) numpy.ndarray
        The mask.
    """
    coeffs = {'dilate': (int(np.ceil(8/dt)), 200)}

    mask = image > 2.5/np.sqrt(dt)

    # Dilate the mask to find approximate areas with highest values
    kernel = np.ones(coeffs['dilate'], np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel)

    # Invert ones and zeros to get the correct mask
    mask = np.logical_not(mask)

    return mask


def fit_mask(image, fit_func, dt, bandwidth):
    """
    Create a mask of the spectrogram around a fitting function.

    The mask only keeps the parts of the image that are within a certain width
    around the curve defined by the fitting function.

    Parameters
    ----------
    image : (N, M) numpy.ndarray
        The image/spectrogram to be masked.
    fit_func : func
        The fitting function to mask around.
    dt : int or float
        The timestep of the spectrogram.
        Only used to give an initial guess of the coefficients.
    bandwidth : float or int
        The width of the band around the fitted curve.

    Returns
    -------
    (N, M) numpy.ndarray
        The mask.
    """
    mask = np.arange(image.shape[1])
    mask = np.broadcast_to(mask, image.shape)
    times = np.arange(image.shape[0]) * dt
    fits = fit_func(times)
    mask = abs(mask - np.reshape(fits, (-1, 1)))

    mask = mask < bandwidth/2

    return mask


def time_mask(image, start_time, end_time, dt):
    """
    Create a mask that removes the data before and after the visible pass.

    Parameters
    ----------
    image : (N, M) numpy.ndarray
        The image/spectrogram to be masked.
    start_time : float
        The start of the pass in seconds since beginning of recording.
    end_time : float
        The end of the pass in seconds since beginning of recording.
    dt : int or float
        The timestep of the spectrogram.
        Only used to give an initial guess of the coefficients.

    Returns
    -------
    (N, M) numpy.ndarray
        The mask.
    """
    min_index = int(start_time / dt)
    max_index = int(end_time / dt)
    mask = np.zeros(image.shape)
    mask[min_index:max_index, :] = True

    return mask

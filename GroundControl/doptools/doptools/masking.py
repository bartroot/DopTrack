import numpy as np
import cv2


# These coeffs appear to work well for dt's of both 0.2, 0.5, 1, and 2 seconds
coeffs = {'freq': {'thresh': 0.5,
                   'erode': (1, 80),
                   'dilate': (10, 2000)},
          'time': {'thresh': 1.0,
                   'erode': (20, 1),
                   'dilate': (500, 20)}}


def area_mask(image, ori):

    # TODO add warning for undefined dt

    mask = image > np.mean(image) + coeffs[ori]['thresh'] * np.std(image)

    # Erode the mask to find the location of areas with highest values
    kernel = np.ones(coeffs[ori]['erode'], np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel)

    # Dilate the mask to find approximate areas with highest values
    kernel = np.ones(coeffs[ori]['dilate'], np.uint8)
    mask = cv2.dilate(mask, kernel)

    # Invert ones and zeros to get the correct mask
    mask = np.logical_not(mask)

    return mask


def fit_mask(image, fit_func, dt):

    mask = np.arange(image.shape[1])
    mask = np.broadcast_to(mask, image.shape)
    times = np.arange(image.shape[0]) * dt
    fits = fit_func(times)
    mask = abs(mask - np.reshape(fits, (-1, 1)))

    mask = mask < 0.5 * 600

    return mask


def time_mask(image, start_time, end_time, dt):

    min_index = int(start_time / dt)
    max_index = int(end_time / dt)
    mask = np.zeros(image.shape)
    mask[min_index:max_index, :] = True

    return mask

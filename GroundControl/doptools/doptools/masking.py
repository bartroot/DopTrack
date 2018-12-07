import numpy as np
import cv2


def vertical_mask(image, dt):

    scale_factor = np.sqrt(dt)
    coeffs = {'thresh': 0.5,
              'erode': (int(15/scale_factor), 1),
              'dilate': (int(500/scale_factor), 10)}

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

    scale_factor = np.sqrt(dt)
    coeffs = {'thresh': 0.5,
              'erode': (1, 80),
              'dilate': (int(3/scale_factor), 2000)}

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

    scale_factor = np.sqrt(dt)
    coeffs = {'dilate': (int(10/scale_factor), 50)}

    mask = image > 1

    # Dilate the mask to find approximate areas with highest values
    kernel = np.ones(coeffs['dilate'], np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel)

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

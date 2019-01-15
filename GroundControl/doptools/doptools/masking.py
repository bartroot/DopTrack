import numpy as np
import cv2


def vertical_mask(image, dt):

    coeffs = {'thresh': 0,
              'erode': (int(np.ceil(50/np.sqrt(dt))), 1),
              'dilate': (int(np.ceil(100/dt)), 15)}

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

    coeffs = {'thresh': 5.0,
              'erode': (int(np.ceil(1)), 100),
              'dilate': (int(np.ceil(3/dt)), 1000)}

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

    coeffs = {'dilate': (int(np.ceil(8/dt)), 200)}

    mask = image > 2.5/np.sqrt(dt)#np.mean(image) + 100 * np.std(image)

    # Dilate the mask to find approximate areas with highest values
    kernel = np.ones(coeffs['dilate'], np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel)

    # Invert ones and zeros to get the correct mask
    mask = np.logical_not(mask)

    return mask


def fit_mask(image, fit_func, dt, bandwidth):

    mask = np.arange(image.shape[1])
    mask = np.broadcast_to(mask, image.shape)
    times = np.arange(image.shape[0]) * dt
    fits = fit_func(times)
    mask = abs(mask - np.reshape(fits, (-1, 1)))

    mask = mask < bandwidth/2

    return mask


def time_mask(image, start_time, end_time, dt):

    min_index = int(start_time / dt)
    max_index = int(end_time / dt)
    mask = np.zeros(image.shape)
    mask[min_index:max_index, :] = True

    return mask

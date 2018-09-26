import numpy as np
import cv2


coeffs = {1: {'freq': {'tresh': 1.0,
                       'erode': (1, 40),
                       'dilate': (4, 2000)},
              'time': {'tresh': 1.0,
                       'erode': (10, 1),
                       'dilate': (50, 5)}},
          0.2: {'freq': {'tresh': 0.5,
                         'erode': (1, 80),
                         'dilate': (10, 2000)},
                'time': {'tresh': 1.0,
                         'erode': (20, 1),
                         'dilate': (500, 20)}}
          }


def area_mask(image, ori, dt):

    mask = image > np.mean(image) + coeffs[dt][ori]['tresh'] * np.std(image)

    # Erode the mask to find the location of areas with highest values
    kernel = np.ones(coeffs[dt][ori]['erode'], np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel)

    # Dilate the mask to find approximate areas with highest values
    kernel = np.ones(coeffs[dt][ori]['dilate'], np.uint8)
    mask = cv2.dilate(mask, kernel)

    # Invert ones and zeros to get the correct mask
    mask = np.logical_not(mask)

    return mask

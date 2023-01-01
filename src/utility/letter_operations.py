from typing import Union

import numpy as np
from PIL import Image
import cv2
import numpy


def rotate(matrix: numpy.ndarray, angle: int) -> numpy.ndarray:
    """
    Rotates provided matrix array by specified angle.
    :param matrix: numpy.ndarray
    :param angle: int
    :return:
    """
    img = Image.fromarray(matrix)
    img = img.rotate(angle)
    return numpy.array(img)


def rotate_cv2(matrix: numpy.ndarray, angle: int) -> numpy.ndarray:
    """
    Rotates provided matrix array by specified angle.
    :param matrix: numpy.ndarray
    :param angle: int
    :return:
    """
    h, w = matrix.shape[0], matrix.shape[1]
    center_x, center_y = w // 2, h // 2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    return cv2.warpAffine(matrix, M, (w, h))


def stretch(matrix: numpy.ndarray, ratio: float, axis: int = 1) -> numpy.ndarray:
    """
    Stretches provided matrix array by specified ratio. Ratio expected as decimal representation of percent (eg. 1.30 for 130%)
    :param matrix: numpy.ndarray
    :param ratio: float
    :param axis: int
    :return: numpy.ndarray
    """
    img = Image.fromarray(matrix)
    if axis == 0:
        new_size = (int(32 * ratio), 32)
    elif axis == 1:
        new_size = (32, int(32 * ratio))
    else:
        new_size = (int(32 * ratio), int(32 * ratio))
    img = img.resize(new_size)
    if axis == 0:
        if ratio > 1:
            img = img.crop((2, 0, 34, 32))
        else:
            img = img.crop((-2, 0, 30, 32))
    elif axis == 1:
        if ratio > 1:
            img = img.crop((0, 2, 32, 34))
        else:
            img = img.crop((0, -2, 32, 30))
    else:
        if ratio > 1:
            img = img.crop((2, 2, 34, 34))
        else:
            img = img.crop((-2, -2, 30, 30))
    return numpy.array(img)


def stretch_cv2(matrix: numpy.ndarray, ratio: float, axis: int = 1) -> numpy.ndarray:
    """
    Stretches provided matrix array by specified ratio. Ratio expected as decimal representation of percent (eg. 1.30 for 130%)
    :param matrix: numpy.ndarray
    :param ratio: float
    :param axis: int
    :return: numpy.ndarray
    """

    y_shape, x_shape = matrix.shape[0], matrix.shape[1]

    stretched_y_shape = int(y_shape * ratio)
    stretched_x_shape = int(x_shape * ratio)

    diff_y_shape = y_shape - stretched_y_shape
    diff_x_shape = x_shape - stretched_x_shape

    if axis == 0:
        new_size = (stretched_y_shape, x_shape)
    elif axis == 1:
        new_size = (y_shape, stretched_x_shape)
    else:
        new_size = (stretched_y_shape, stretched_x_shape)

    resized = cv2.resize(matrix, new_size)

    if axis == 0:
        if ratio > 1:
            img = resized[0:y_shape,
                          int((stretched_x_shape / 2) - (x_shape / 2)):int((stretched_x_shape / 2) + (x_shape / 2))]
        else:
            if diff_x_shape % 2 == 0:
                img = numpy.pad(resized,
                                ((0, 0),
                                 (int(diff_x_shape / 2), int(diff_x_shape / 2))),
                                constant_values=0)
            else:
                img = numpy.pad(resized,
                                ((0, 0),
                                 (int((diff_x_shape / 2 + 1)), int(diff_x_shape / 2))),
                                constant_values=0)
    elif axis == 1:
        if ratio > 1:
            img = resized[int((stretched_y_shape / 2) - (y_shape / 2)):int((stretched_y_shape / 2) + (y_shape / 2)),
                  0:x_shape]
        else:
            if diff_y_shape % 2 == 0:
                img = numpy.pad(resized,
                                ((int(diff_y_shape / 2), int(diff_y_shape / 2)),
                                 (0, 0)),
                                constant_values=0)
            else:
                img = numpy.pad(resized,
                                ((int((diff_y_shape / 2 + 1)), int(diff_y_shape / 2)),
                                 (0, 0)),
                                constant_values=0)

    else:
        if ratio > 1:
            img = resized[int((stretched_y_shape / 2) - (y_shape / 2)):int((stretched_y_shape / 2) + (y_shape / 2)),
                          int((stretched_x_shape / 2) - (x_shape / 2)):int((stretched_x_shape / 2) + (x_shape / 2))]
        else:
            if diff_x_shape % 2 == 0:
                img = numpy.pad(resized,
                                ((int(diff_y_shape / 2), int(diff_y_shape / 2)),
                                 (int(diff_x_shape / 2), int(diff_x_shape / 2))),
                                constant_values=0)
            else:
                img = numpy.pad(resized,
                                ((int(diff_y_shape / 2 + 1), int(diff_y_shape / 2)),
                                 (int(diff_x_shape / 2 + 1), int(diff_x_shape / 2))),
                                constant_values=0)

    return numpy.array(img)


def thresholding(matrix: numpy.ndarray, threshold: Union[int, float]) -> numpy.ndarray:
    """
    Performs threshholding on provided matrix array.
    :param matrix: numpy.ndarray
    :param threshold: int or float
    :return: numpy.ndarray
    """
    return numpy.where(matrix >= threshold, 255, 0)


def invert_pixels(matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Inverts provided matrix array. Notice that matrix must be of type numpy.uint8
    :param matrix: numpy.ndarray
    :return: numpy.ndarray
    """
    return 255 - matrix if matrix.max() == 255 else 1 - matrix

def noise(matrix: numpy.ndarray, noise_amount: float) -> numpy.ndarray:
    """
    This function wraps add_noise, and it's kept for backward compatibility.
    """
    return add_noise(matrix, noise_amount / 255) # noise_amount is in range 0-255, but add_noise expects 0-1


def add_noise(matrix: numpy.ndarray, noise_amount: float) -> numpy.ndarray:
    """
    Adding noise on provided matrix array.
    :param matrix: numpy.ndarray
    :param noise_amount: float from 0 to 1 (less is more noise)
    :return: numpy.ndarray (int8)
    """
    # add image with generated thresholded noise, then clip values to [0, 255]
    return thresholding(
        (matrix + thresholding(np.random.normal(0, 0.5, size=matrix.shape), noise_amount)),
        255
    )
import cv2
import numpy as np


def dilation(image: np.ndarray, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def dilation_elipse(image: np.ndarray, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.dilate(image, kernel, iterations=1)

import cv2
import numpy as np


def morphological_gradient(image: np.ndarray, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

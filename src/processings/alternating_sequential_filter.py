import cv2
import numpy as np


def structuring_element(i: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (2 * i + 1, 2 * i + 1))


def morphological_close(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(image, kernel)
    return cv2.erode(dilated, kernel)


def morphological_open(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(image, kernel)
    return cv2.dilate(eroded, kernel)


def alternating_sequential_filter(image: np.ndarray, seq: str, n: int) -> np.ndarray:
    y = image
    if seq == "OC":
        for i in range(1, n + 1):
            nb = structuring_element(i)
            y = morphological_close(y, nb)
            y = morphological_open(y, nb)
    elif seq == "CO":
        for i in range(1, n + 1):
            nb = structuring_element(i)
            y = morphological_open(y, nb)
            y = morphological_close(y, nb)
    return y

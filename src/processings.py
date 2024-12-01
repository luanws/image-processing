import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion, label
from skimage import morphology


def dilation(image: np.ndarray, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def dilation_elipse(image: np.ndarray, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.dilate(image, kernel, iterations=1)


def erosion(image: np.ndarray, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def erosion_elipse(image: np.ndarray, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.erode(image, kernel, iterations=1)


def conditional_erosion(
    marker: np.ndarray,
    mask: np.ndarray,
    structuring_element: np.ndarray,
) -> np.ndarray:
    prev_marker: np.ndarray = np.zeros_like(marker)
    while not np.array_equal(marker, prev_marker):
        prev_marker = marker.copy()
        eroded: np.ndarray = cv2.erode(marker, structuring_element)
        marker = np.maximum(eroded, mask)
    return marker


def morphological_gradient(image: np.ndarray, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def remove_small_regions(image: np.ndarray, min_size: int) -> np.ndarray:
    labels, number_of_labels = label(image)
    count = np.bincount(labels.flatten())[1:]
    mask = np.zeros_like(labels, dtype=np.bool_)
    for i in range(1, number_of_labels + 1):
        if count[i - 1] >= min_size:
            mask |= labels == i
    image_without_stain = np.zeros_like(image)
    image_without_stain[mask] = image[mask]
    return image_without_stain


def toggle_highlight(image: np.ndarray, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


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


def reconstruction_by_dilation(image: np.ndarray, marker: np.ndarray) -> np.ndarray:
    marker = np.minimum(marker, image)
    reconstructed = morphology.reconstruction(marker, image, method="dilation")
    return reconstructed


def reconstruction_by_erosion(image: np.ndarray, marker: np.ndarray) -> np.ndarray:
    marker = np.maximum(marker, image)
    reconstructed = morphology.reconstruction(marker, image, method="erosion")
    return reconstructed


def h_maxima_transform(image: np.ndarray, h: float, footprint=(3, 3)) -> np.ndarray:
    elevated_image = image + h
    reconstructed = grey_dilation(
        np.minimum(elevated_image, image.max()), footprint=np.ones(footprint)
    )
    return reconstructed - h


def h_minima_transform(image: np.ndarray, h: float, footprint=(3, 3)) -> np.ndarray:
    lowered_image = image - h
    reconstructed = grey_erosion(
        np.maximum(lowered_image, image.min()), footprint=np.ones(footprint)
    )
    return reconstructed + h


def remove_stain(image: np.ndarray, min_size: int) -> np.ndarray:
    labels, number_of_labels = label(image)
    count = np.bincount(labels.flatten())[1:]
    mask = np.zeros_like(labels, dtype=np.bool_)
    for i in range(1, number_of_labels + 1):
        if count[i - 1] >= min_size:
            mask |= labels == i
    image_without_stain = np.zeros_like(image)
    image_without_stain[mask] = image[mask]
    return image_without_stain


def heaviside(image: np.ndarray, threshold: float) -> np.ndarray:
    return np.heaviside(image - threshold, 1)


def enhance_contrast(image: np.ndarray, size: int = 4, iterations=5) -> np.ndarray:
    kernel = np.ones((size, size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    eroded = cv2.erode(dilated, kernel, iterations=iterations)
    return eroded

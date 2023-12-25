"""
Apply thresholding to the input image to segment the lung region.
"""
import numpy as np
import cv2 as cv


def segment_with_threshold(image: np.ndarray, threshold: float = 100) -> np.ndarray:
    colored_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, segmented_image = cv.threshold(colored_image, threshold, 255, cv.THRESH_BINARY)
    return segmented_image

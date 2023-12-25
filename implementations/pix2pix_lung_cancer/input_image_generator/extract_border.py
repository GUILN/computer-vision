"""
Extracts the border of the lung from the image - so that it can be used as the 
input image for the pix2pix model.
"""
import numpy as np
import cv2 as cv


def extract_border(
    image: np.ndarray,
    canny_low: float = 128,
    threshold: float = 100,
) -> np.ndarray:
    """
    Extracts the border of the lung from the image.
    Apply Canny edge detection to the image to extract the border of the lung.
    :param image: the image
    :return: the border of the lung
    """
    edges = cv.Canny(image, canny_low, threshold)
    return edges

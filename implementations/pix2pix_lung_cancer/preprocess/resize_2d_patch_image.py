"""
Resize 2D patch image to 128x128 - as in the paper
"""
from scipy import ndimage
import numpy as np

def resize(image: np.ndarray, factor=2):
    # Assuming 'image' is your numpy array
    zoom_factor = factor  # 2 means doubling the size
    resized_image = ndimage.zoom(image, zoom_factor)
    return resized_image

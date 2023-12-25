from collections import namedtuple
import numpy as np
import cv2 as cv
import logging
from typing import Tuple
from input_image_generator.extract_border import extract_canny_border
from input_image_generator.segment_image import segment_with_threshold


BorderExtractionParams = namedtuple(
    "BorderExtractionParams", ["canny_low", "threshold"]
)


class InputImageGenerator:
    """
    Input image generator for pix2pix model.
    The idea is that we will simulate free form drawing by
    using the border of the lung as the input image.
    """

    def __init__(
        self,
        resize: Tuple[int, int] = (128, 128),
        segmentation_threshold: float = 100,
        border_extraction_params: BorderExtractionParams = BorderExtractionParams(
            128, 100
        ),
    ):
        """
        param segmentation_threshold: the threshold to use for segmentation
        param border_extraction_params: the parameters to use for border extraction
        """
        logging.debug("Initializing InputImageGenerator...")
        self._segmentation_threshold = segmentation_threshold
        self._border_extraction_params = border_extraction_params
        self._resize = resize

    def generate_input_image(self, image: np.ndarray) -> np.ndarray:
        """
        Generate input images for pix2pix model.
        :param image: the image
        :return: the input image
        """
        logging.debug("Generating input images...")
        logging.debug("Applying thresholding to the image...")
        segmented_image = segment_with_threshold(image)
        logging.debug("Extracting border from the image...")
        border = extract_canny_border(segmented_image)
        logging.debug("Resizing the image...")
        border = cv.resize(border, self._resize)
        logging.debug("Done generating input images.")
        return border

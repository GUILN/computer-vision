from collections import namedtuple
import numpy as np
import logging

from extract_border import extract_border
from segment_image import segment_image

BorderExtractionParams = namedtuple(
    "BorderExtractionParams", ["canny_low", "threshold"]
)


class InputImageGenerator:
    def __init__(
        self,
        segmentation_threshold: float = 100,
        border_extraction_params: BorderExtractionParams = BorderExtractionParams(
            128, 100
        ),
    ):
        logging.debug("Initializing InputImageGenerator...")
        self._segmentation_threshold = segmentation_threshold
        self.border_extraction_params = border_extraction_params

    def generate_input_image(self, image: np.ndarray) -> np.ndarray:
        """
        Generate input images for pix2pix model.
        :param image: the image
        :return: the input image
        """
        logging.debug("Generating input images...")
        logging.debug("Applying thresholding to the image...")
        segmented_image = segment_image(image)
        logging.debug("Extracting border from the image...")
        border = extract_border(segmented_image)
        logging.debug("Done generating input images.")
        return border

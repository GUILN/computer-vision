from collections import namedtuple
import logging
from typing import List, Tuple

import cv2 as cv
import numpy as np

from input_image_generator.generator import InputImageGenerator

TestImageTuple = namedtuple("TestData", ["input_image", "image"])


class TestDataPipeline:
    def __init__(
        self,
        resize: Tuple[int, int] = (128, 128),
        rotation_augmentation_degree: int = None,
    ):
        self._rotation_augmentation_degree = rotation_augmentation_degree
        self._input_image_generator = InputImageGenerator()
        self._resize = resize

    @property
    def total_images_per_test_image(self) -> int:
        """
        Returns the total number of images per test image.
        :return: the total number of images per test image
        """
        return 360 // self._rotation_augmentation_degree

    def generate_test_image(self, image: np.ndarray) -> List[TestImageTuple]:
        """
        Generate input images for pix2pix model.
        Returns a list depending on the number of augmentation degrees.
        :param image: the image
        :return: the input image
        """
        logging.debug("Generating input images...")
        logging.debug(
            "generating %d images per test image", self.total_images_per_test_image
        )
        images: List[TestImageTuple] = []
        for img in self._get_all_rotated_images(image):
            input_image = cv.resize(
                self._input_image_generator.generate_input_image(img), self._resize
            )
            img = cv.resize(img, self._resize)
            images.append(
                TestImageTuple(
                    input_image=input_image,
                    image=img,
                )
            )
        return images

    def _get_all_rotated_images(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Returns a list of all rotated images.
        :param image: the image
        :return: the list of all rotated images
        """
        images = [image]
        rows, cols = image.shape
        if self._rotation_augmentation_degree:
            for i in range(0, 360, self._rotation_augmentation_degree):
                M = cv.getRotationMatrix2D((cols / 2, rows / 2), i, 1)
                rotated = cv.warpAffine(image, M, (cols, rows))
                images.append(rotated)
        return images

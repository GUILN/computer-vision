#!/usr/bin/env python3
import logging
import argparse
from argparse import Namespace
from typing import Callable, Generator, Tuple
import os

import cv2

from test_data_pipeline import TestDataPipeline, TestImageTuple


def parse_args() -> Namespace:
    """
    returns the parsed args
    """
    parset = argparse.ArgumentParser()
    parset.add_argument(
        "--save_dir",
        type=str,
        default="./data/test_saved_input_data",
        help="data directory",
    )
    parset.add_argument(
        "--rotate_images",
        type=int,
        default=30,
        help="Angle to rotate images in order to create more images",
    )
    parset.add_argument(
        "--patches_dir",
        type=str,
        default="./data/LUNA_patches",
        help="patches directory",
    )

    return parset.parse_args()


def get_test_data(
    patches_dir: str,
    rotate_images: int,
    extra_processing: Callable[[TestImageTuple], TestImageTuple] = None,
) -> Generator[Tuple[TestImageTuple, str], None, None]:
    """
    Returns a generator of test data.
    :param data_dir: the data directory
    :param rotate_images: the number of degrees to rotate images
    :return: a generator of test data and the image name
    """
    test_data_pipeline = TestDataPipeline(
        rotation_augmentation_degree=rotate_images, resize=(256, 256)
    )
    # get only the first image from the data directory
    logging.info("Getting test data...")
    logging.info("Total images: %d", len(os.listdir(patches_dir)))
    logging.info(
        "Total images to be generated: %d",
        len(os.listdir(patches_dir)) * test_data_pipeline.total_images_per_test_image,
    )
    for file in os.listdir(patches_dir):
        if file.endswith(".png"):
            image_name = file
            image_path = os.path.join(patches_dir, image_name)
            image = cv2.imread(image_path)
            for img_i, test_image in enumerate(
                test_data_pipeline.generate_test_image(image)
            ):
                if extra_processing is not None:
                    logging.debug("Applying extra processing...")
                    test_image = extra_processing(test_image)
                image_complete_name = f"{image_name}_{img_i}"
                yield test_image, image_complete_name


def main():
    logging.info("Parsing args...")
    args = parse_args()
    logging.info("Processing with arguments: %s", str(args))
    logging.info("Getting test data...")
    for test_image, image_name in get_test_data(args.patches_dir, args.rotate_images):
        input_image = cv2.bitwise_not(test_image.input_image)
        # expand dims
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        logging.info(
            "Images sizes - input: %s, real: %s",
            str(test_image.input_image.shape),
            str(test_image.image.shape),
        )
        # concatenate images
        concatenated_image = cv2.hconcat([input_image, test_image.image])
        # save image
        cv2.imwrite(
            os.path.join(args.save_dir, f"{image_name}.png"), concatenated_image
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()

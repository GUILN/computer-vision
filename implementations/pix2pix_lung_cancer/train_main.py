#!/usr/bin/env python3
import logging
import argparse
from argparse import Namespace
import os

import cv2

from test_data_pipeline import TestDataPipeline

TEST_SAVED_INPUT_DATA = "./data/test_saved_input_data"


def parse_args() -> Namespace:
    """
    returns the parsed args
    """
    parset = argparse.ArgumentParser()
    parset.add_argument("--data_dir", type=str, default="./data", help="data directory")
    parset.add_argument(
        "--rotate_images",
        type=int,
        default=30,
        help="Angle to rotate images in order to create more images",
    )
    parset.add_argument(
        "--patches_dir", type=str, default="./data/patches", help="patches directory"
    )
    parset.add_argument(
        "--save_model_dir",
        type=str,
        default=None,
        help="directory to save the model",
    )
    parset.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train"
    )

    return parset.parse_args()


def get_test_data(data_dir: str, rotate_images: int):
    test_data_pipeline = TestDataPipeline(rotation_augmentation_degree=rotate_images)
    # get only the first image from the data directory
    image_name = os.listdir(data_dir)[0]
    image_path = os.path.join(data_dir, image_name)
    # read image
    image = cv2.imread(image_path)
    test_data = test_data_pipeline.generate_test_image(image)
    for i, test_image in enumerate(test_data):
        # save image with cmap gray_r
        cv2.imwrite(
            os.path.join(TEST_SAVED_INPUT_DATA, f"input_{i}.png"),
            cv2.bitwise_not(test_image.input_image),
        )
        # save image
        cv2.imwrite(
            os.path.join(TEST_SAVED_INPUT_DATA, f"real_image_{i}.png"), test_image.image
        )


def main():
    logging.info("Parsing args...")
    args = parse_args()
    logging.info("Processing with arguments: %s", str(args))
    logging.info("Getting test data...")
    get_test_data(args.data_dir, args.rotate_images)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
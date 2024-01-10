#!/usr/bin/env python3

import argparse
import logging
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

from input_image_generator.extract_border import extract_canny_border
from gan_network.pix2pix_data_pipeline import load_production_input_image
from gan_network.gan_model import GanModel


def pre_process_sketch(
    sketch_file: str, resize_factor: Tuple[int, int] = (400, 400)
) -> np.ndarray:
    logging.info("Pre-processing sketch file: %s", sketch_file)
    logging.info("Reading sketch file...")
    image = cv2.imread(sketch_file)
    logging.info("Applying Cany edge detection...")
    border_file = extract_canny_border(image, canny_low=128, threshold=200)
    logging.info("Resizing border file...")
    border_file = cv2.resize(border_file, resize_factor)
    logging.info("Inverting border file...")
    border_file = cv2.bitwise_not(border_file)
    border_file = cv2.cvtColor(border_file, cv2.COLOR_GRAY2BGR)
    return border_file


def parse_args() -> argparse.Namespace:
    parsert = argparse.ArgumentParser()
    parsert.add_argument("--out", type=str, default="./out.png", help="Output file")
    parsert.add_argument("--sketch", type=str, help="Sketch file to be processed")
    return parsert.parse_args()


def main():
    # set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    args = parse_args()
    # assert sketch is a png image
    assert (
        args.sketch.endswith(".png")
        or args.sketch.endswith(".jpg")
        or args.sketch.endswith(".jpeg")
    )
    logging.info("Arguments: %s", str(args))
    pre_processed_sketch = pre_process_sketch(args.sketch)
    if args.sketch.endswith(".png"):
        input_image_file_name = args.sketch.split(".png")[0] + "_input.png"
    elif args.sketch.endswith(".jpg"):
        input_image_file_name = args.sketch.split(".jpg")[0] + "_input.png"
    elif args.sketch.endswith(".jpeg"):
        input_image_file_name = args.sketch.split(".jpeg")[0] + "_input.png"
    logging.info(
        "saving pre-processed sketch to file (used as input image): %s",
        input_image_file_name,
    )
    cv2.imwrite(input_image_file_name, pre_processed_sketch)
    input_image_treated = load_production_input_image(input_image_file_name)
    logging.info("Loading GAN model...")
    CHKPT_DIR = "./data/checkpoints_2"
    LOG_DIR = "./data/logs"
    GENERATED_IMGS_DIR = "../data/saved_input_data_test_2_results"
    gan_model = GanModel(
        checkpoint_dir=CHKPT_DIR,
        save_image_dir=GENERATED_IMGS_DIR,
        log_dir=LOG_DIR,
        load_checkpoint=True,
    )
    logging.info("Generating images...")
    gan_model.generate_image(input_image_treated[tf.newaxis, ...], args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import logging
from typing import Tuple

import cv2
import numpy as np

from input_image_generator.extract_border import extract_canny_border


def pre_process_sketch(
    sketch_file: str, resize_factor: Tuple[int, int] = (256, 256)
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
    assert args.sketch.endswith(".png") or args.sketch.endswith(".jpg")
    logging.info("Arguments: %s", str(args))
    pre_processed_sketch = pre_process_sketch(args.sketch)
    if args.sketch.endswith(".png"):
        input_image_file_name = args.sketch.split(".png")[0] + "_input.png"
    else:
        input_image_file_name = args.sketch.split(".jpg")[0] + "_input.png"
    logging.info(
        "saving pre-processed sketch to file (used as input image): %s",
        input_image_file_name,
    )
    cv2.imwrite(input_image_file_name, pre_processed_sketch)


if __name__ == "__main__":
    main()

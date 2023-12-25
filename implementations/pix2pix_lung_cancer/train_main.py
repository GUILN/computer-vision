#!/usr/bin/env python3
import logging
import argparse
from argparse import Namespace


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


def main():
    logging.info("Parsing args...")
    args = parse_args()
    logging.info("Processing with arguments: %s", str(args))
    logging.info("Getting test data...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()

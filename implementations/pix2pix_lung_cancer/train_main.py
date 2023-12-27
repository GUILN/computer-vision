#!/usr/bin/env python3
import logging
import argparse
from argparse import Namespace
from pix2pix_data_pipeline import get_train_dataset


def parse_args() -> Namespace:
    """
    returns the parsed args
    """
    parset = argparse.ArgumentParser()
    parset.add_argument(
        "--save_model",
        type=str,
        default="./lung_cancer_pix2pix_model.model",
        help="save model directory",
    )
    parset.add_argument(
        "--input_data_dir",
        type=str,
        default="./data/saved_input_data",
        help="input data directory (previously generated, rotated images)",
    )
    parset.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs",
    )
    parset.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer",
    )
    parset.add_argument(
        "--learning_rate",
        type=float,
        default=0.00002,
        help="learning rate",
    )

    return parset.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Parsing args...")
    args = parse_args()
    logging.info("Processing with arguments: %s", str(args))
    logging.info("Getting test data...")
    train_dataset = get_train_dataset(args.input_data_dir)
    logging.info("Training dataset: %s", str(train_dataset))
    logging.info("Training dataset length: %d", len(train_dataset))
    for img in train_dataset.take(1):
        # log the image
        logging.info("Image: %s", str(img))
    logging.info("Training model...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise e

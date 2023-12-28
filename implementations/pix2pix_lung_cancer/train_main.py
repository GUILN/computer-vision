#!/usr/bin/env python3
import logging
import argparse
from argparse import Namespace
from gan_network.pix2pix_data_pipeline import get_test_dataset, get_train_dataset
from gan_network.gan_model import GanModel


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
        "--train_data_dir",
        type=str,
        default="./data/saved_input_data",
        help="input data directory (previously generated, rotated images)",
    )
    parset.add_argument(
        "--test_data_dir",
        type=str,
        default="./data/saved_input_data_test",
        help="input data directory (previously generated, rotated images)",
    )
    parset.add_argument(
        "--log_dir",
        type=str,
        default="./data/logs",
        help="logs dir",
    )
    parset.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./data/checkpoint",
        help="checkpoint dir - to save the model",
    )
    parset.add_argument(
        "--generated_images_dir",
        type=str,
        default="./data/generated_images",
        help="generated images dir - to save the generated images while training",
    )
    parset.add_argument(
        "--steps",
        type=int,
        default=40000,
        help="number of steps",
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
    logging.info("Getting train data...")
    train_dataset = get_train_dataset(args.train_data_dir)
    logging.info("Getting test data...")
    test_dataset = get_test_dataset(args.test_data_dir)
    logging.info("Initializing model...")
    model = GanModel(
        checkpoint_dir=args.checkpoint_dir,
        save_image_dir=args.generated_images_dir,
        log_dir=args.log_dir,
    )
    logging.info("Starting training (fitting)...")
    model.fit(train_dataset, test_dataset, steps=args.steps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise e

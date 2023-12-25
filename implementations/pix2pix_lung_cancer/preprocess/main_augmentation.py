import argparse
import logging
import os

DEFAULT_DEGREES = 30

def process_images(data_dir: str, rotate_degrees: int):
    """
    Process images.
    """
    logging.info("Processing images in directory: %s", data_dir)
    files = os.listdir(data_dir)
    logging.info("Processing %d files...", len(files))
    total_num_rotations = (360 // rotate_degrees) - 1
    logging.info("Rotating images to %d degrees...", rotate_degrees)
    logging.info(
        "Total number of rotations: %d ((div 360) - 1)",
        total_num_rotations
    )
    logging.info("Total number of images after augmentation: %d", len(files) * total_num_rotations)


def main():
    # parse args
    parsert = argparse.ArgumentParser()
    parsert.add_argument('--data_dir', type=str, help='data dir to with images to augment')
    parsert.add_argument('--rotate_degrees', type=int, default=DEFAULT_DEGREES, help=f'degrees to roate images in order to create more images - default {DEFAULT_DEGREES} degrees')
    args = parsert.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Start preprocessing...")
    logging.info("Arguments: %s", str(args))
    process_images(args.data_dir, args.rotate_degrees)
 
  
if __name__ == "__main__":
    main()
    
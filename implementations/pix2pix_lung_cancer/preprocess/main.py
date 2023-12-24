#!/usr/bin/env python3
import os
import argparse
import logging
from tqdm import tqdm


def process_file(file_path: str, patches_dir: str):
    logging.info("Processing file: %s", file_path)
    logging.info("extracting rois for file: %s", file_path)
    logging.info("resizing patches for file: %s", file_path)
    logging.info("saving patches for file: %s", file_path)
    

def preprocess_pipeline(data_dir: str, patches_dir: str):
    all_files = os.listdir(data_dir)
    # filter mhd files
    all_files = list(filter(lambda x: x.endswith('.mhd'), all_files))
    logging.info("Total files: %d", len(all_files))
    for file in tqdm(all_files):
        process_file(file, patches_dir)


def main():
    # parse args
    parsert = argparse.ArgumentParser()
    parsert.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parsert.add_argument('--patches_dir', type=str, default='./data/patches', help='patches directory')
    args = parsert.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Start preprocessing...")
    logging.info("Arguments: %s", str(args))
    preprocess_pipeline(args.data_dir, args.patches_dir)


if __name__ == "__main__":
    main()
    
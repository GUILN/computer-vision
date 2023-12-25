#!/usr/bin/env python3
import os
import argparse
import logging
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from extract_roi_to_2d_patch import extract_candidates
from resize_2d_patch_image import resize


def process_file(file_path: str, patches_dir: str, annotations_df: pd.DataFrame, candidates_df: pd.DataFrame):
    logging.info("Processing file: %s", file_path)
    logging.info("extracting rois for file: %s", file_path)
    patches_array, values_array, nodule_diameters_array = extract_candidates(file_path, candidates_df, annotations_df, 20)
    logging.info("filtering positive nodules...")
    num_positives = np.where(values_array == 1)[0]
    logging.info("total positive nodules: %d", len(num_positives))
    for i, candidate_num in enumerate(num_positives):
        logging.info("processing positive nodule: %d", candidate_num)
        resized = resize(patches_array[candidate_num], factor=10)
        logging.info("resizing patch...")
        resized_2 = cv2.resize(resized, (128, 128))
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        fig_name = os.path.join(patches_dir, file_name + "_" + str(i) + '.png')
        logging.info("saving patch: %s", fig_name)
        plt.imsave(fig_name, resized_2) 
        
    logging.info("resizing patches for file: %s", file_path)
    logging.info("saving patches for file: %s", file_path)
    

def preprocess_pipeline(data_dir: str, patches_dir: str, annotations_df: pd.DataFrame, candidates_df: pd.DataFrame):
    all_files = os.listdir(data_dir)
    # filter mhd files
    all_files = list(filter(lambda x: x.endswith('.mhd'), all_files))
    logging.info("Total files: %d", len(all_files))
    for file in tqdm(all_files):
        file = os.path.join(data_dir, file)
        process_file(file, patches_dir, annotations_df, candidates_df)


def main():
    # parse args
    parsert = argparse.ArgumentParser()
    parsert.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parsert.add_argument('--patches_dir', type=str, default='./data/patches', help='patches directory')
    parsert.add_argument('--annotations_file', type=str, default='./data/annotations.csv', help='annotations file')
    parsert.add_argument('--candidates_file', type=str, default='./data/candidates.csv', help='candidates file')
    args = parsert.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Start preprocessing...")
    logging.info("Arguments: %s", str(args))
   
    logging.info("Loading annotations file: %s", args.annotations_file)
    annotations_df = pd.read_csv(args.annotations_file)
    logging.info("Loading candidates file: %s", args.candidates_file)
    candidates_df = pd.read_csv(args.candidates_file) 
    
    preprocess_pipeline(args.data_dir, args.patches_dir, annotations_df, candidates_df)


if __name__ == "__main__":
    main()
    
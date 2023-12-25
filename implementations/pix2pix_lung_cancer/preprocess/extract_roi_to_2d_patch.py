"""
Extracts the region of interest from the CT scan.
"""
import ntpath
import pandas as pd
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from os.path import join as path_join
import logging
from typing import Tuple


def normalize_planes(npzarray):
    """
    Normalize pixel depth into Hounsfield units (HU)

    This tries to get all pixels between -1000 and 400 HU.
    All other HU will be masked.

    """
    maxHU = 400.0
    minHU = -1000.0

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray


def extract_candidates(
    img_file: str, candidates: pd.DataFrame, annotations: pd.DataFrame, window: int = 20
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extracts the region of interest from the CT scan.
    :param img_file: the path to the CT scan
    :param window: the window size
    :param candidates: the candidates dataframe
    :param annotations: the annotations dataframe
    :return: the region of interest
    """
    # Get the name of the file
    subjectName = ntpath.splitext(ntpath.basename(img_file))[
        0
    ]  # Strip off the .mhd extension

    # Read the list of candidate ROI
    dfCandidates = candidates.copy()
    dfAnnotations = annotations.copy()

    numCandidates = dfCandidates[dfCandidates["seriesuid"] == subjectName].shape[0]
    logging.info("There are {} candidate nodules in this file.".format(numCandidates))

    numNonNodules = sum(
        dfCandidates[dfCandidates["seriesuid"] == subjectName]["class"] == 0
    )
    numNodules = sum(
        dfCandidates[dfCandidates["seriesuid"] == subjectName]["class"] == 1
    )
    logging.info(
        "{} are true nodules (class 1) and {} are non-nodules (class 0)".format(
            numNodules, numNonNodules
        )
    )

    # Read if the candidate ROI is a nodule (1) or non-nodule (0)
    candidateValues = dfCandidates[dfCandidates["seriesuid"] == subjectName][
        "class"
    ].values

    # Get the world coordinates (mm) of the candidate ROI center
    worldCoords = dfCandidates[dfCandidates["seriesuid"] == subjectName][
        ["coordX", "coordY", "coordZ"]
    ].values

    # Use SimpleITK to read the mhd image
    itkimage = sitk.ReadImage(img_file)

    # Get the real world origin (mm) for this image
    originMatrix = np.tile(
        itkimage.GetOrigin(), (numCandidates, 1)
    )  # Real world origin for this image (0,0)

    # Subtract the real world origin and scale by the real world (mm per pixel)
    # This should give us the X,Y,Z coordinates for the candidates
    candidatesPixels = (
        np.round(np.absolute(worldCoords - originMatrix) / itkimage.GetSpacing())
    ).astype(int)

    # Replace the missing diameters with the 50th percentile diameter

    candidateDiameter = (
        dfAnnotations["diameter_mm"]
        .fillna(dfAnnotations["diameter_mm"].quantile(0.5))
        .values
        / itkimage.GetSpacing()[1]
    )

    candidatePatches = []

    imgAll = sitk.GetArrayFromImage(itkimage)  # Read the image volume

    for candNum in range(numCandidates):
        # print('Extracting candidate patch #{}'.format(candNum))
        candidateVoxel = candidatesPixels[candNum, :]
        xpos = int(candidateVoxel[0])
        ypos = int(candidateVoxel[1])
        zpos = int(candidateVoxel[2])

        # Need to handle the candidates where the window would extend beyond the image boundaries
        windowSize = window
        x_lower = np.max([0, xpos - windowSize])  # Return 0 if position off image
        x_upper = np.min(
            [xpos + windowSize, itkimage.GetWidth()]
        )  # Return  maxWidth if position off image

        y_lower = np.max([0, ypos - windowSize])  # Return 0 if position off image
        y_upper = np.min(
            [ypos + windowSize, itkimage.GetHeight()]
        )  # Return  maxHeight if position off image

        # crop the image for the double of candidate diameter

        # SimpleITK is x,y,z. Numpy is z, y, x.
        imgPatch = imgAll[zpos, y_lower:y_upper, x_lower:x_upper]

        # Normalize to the Hounsfield units
        # TODO: I don't think we should normalize into Housefield units
        imgPatchNorm = normalize_planes(imgPatch)

        candidatePatches.append(
            imgPatchNorm
        )  # Append the candidate image patches to a python list

    return candidatePatches, candidateValues, candidateDiameter


def resize(image, factor=2):
    # Assuming 'image' is your numpy array
    zoom_factor = factor  # 2 means doubling the size
    resized_image = ndimage.zoom(image, zoom_factor)
    return resized_image

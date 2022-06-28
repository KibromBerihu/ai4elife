""" Script to preprocess a given FDG PET image in .nii.
"""
# Import libraries
import glob
import os

import numpy as np
from numpy import ndarray
from numpy.random import seed
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
import warnings

from skimage.transform import resize
import scipy.ndimage
import nibabel as nib
from scipy.ndimage import label

# seed random number generator
seed(1)


def remove_outliers_in_sagittal(predicted):
    """ when the sagittal image based segmentation does not have corresponding image in coronal remove it
    """
    sagittal = np.squeeze(predicted[0, ...])
    coronal = np.squeeze(predicted[1, ...])

    try:
        sagittal = np.squeeze(sagittal)
    except:
        pass

    try:
        coronal = np.squeeze(coronal)
    except:
        pass

    binary_mask = sagittal.copy()
    binary_mask[binary_mask >= 0.5] = 1
    binary_mask[binary_mask < 0.5] = 0
    coronal[coronal >= 0.5] = 1
    coronal[coronal < 0.5] = 0

    labelled_mask, num_labels = label(binary_mask)
    # Let us now remove all the small regions, i.e., less than the specified mm.
    print(binary_mask.shape)
    print(labelled_mask.shape)
    print(coronal.shape)
    for get_label in range(num_labels):
        refined_mask = np.zeros(binary_mask.shape)
        refined_mask[labelled_mask == get_label] = 1
        x, y = np.nonzero(refined_mask)
        x1, y1 = np.max(x), np.min(y)
        x2, y2 = np.min(x), np.max(y)
        refined_mask[:, y1-5:y2+5] = 1
        if (refined_mask * coronal).sum() == 0:
            binary_mask[labelled_mask == get_label] = 0

    predicted[0, ...] = np.expand_dims(binary_mask, axis=-1)

    return predicted


# Read .nii files using itk
if __name__ == '__main__':
    print("remove wrong segmentation on sagittal views")

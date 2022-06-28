"""
By KB Girum
"""
# import libraries
import os
import glob
import nibabel as nib
import imageio
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import h5py

import random
import pandas as pd
from tqdm import tqdm
import cv2

from numpy.random import uniform, exponential
from itertools import cycle
import csv

from lifelines import KaplanMeierFitter
from lifelines.plotting import plot_lifetimes
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from lifelines.plotting import add_at_risk_counts

from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from numpy import ndarray

kmf = KaplanMeierFitter()
from medpy.metric import binary

import ntpath
from scipy.ndimage import label


def superimpose_segmentation_images(pet_gt_prd_display, file_name, logzscore=None):
    """

    Args:
        pet_ct_gt_prd:
        file_name:
        logzscore:
    """
    pet, gt, prd = pet_gt_prd_display[0], pet_gt_prd_display[1], pet_gt_prd_display[2]

    if logzscore == "log":
        pet = np.log(pet + 1)
    elif logzscore == "zscore":
        pet = (pet - np.mean(pet)) / (np.std(pet) + 1e-8)
    elif logzscore == "clipping":
        pet[pet > 50] = 50
        pet /= 50
    else:
        pet = np.log(pet + 1)

    img = pet
    try:
        img = np.squeeze(img, axis=-1)
    except:
        pass
    try:
        gt = np.squeeze(gt, axis=-1)
    except:
        pass

    try:
        prd = np.squeeze(prd, axis=-1)
    except:
        pass

    img = np.rot90(img)

    if len(prd):
        prd = np.rot90(prd)
        prd[prd > 0] = 1

    img = 10 - img
    if len(gt):
        gt = np.rot90(gt)
        gt[gt > 0] = 1

        # miss classified regions
        prd_error = prd + gt
        prd_error[prd_error != 1] = 0
        dice = binary.dc(prd, gt)
        dice = np.round(dice * 100, 1)
    else:
        dice = 'unkown'

    color = ['brg']
    hfont = {'fontname': 'Arial'}
    fontsize_ = 12
    for clr in color:
        viridis = cm.get_cmap(clr)
        print("\n Image ID: \t %s", str(file_name))
        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('PET image', **hfont, fontsize=fontsize_)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])


        axs[1].imshow(img, cmap='gray')
        if len(gt):
            gt = np.ma.masked_where(gt == 0, gt)
            axs[1].imshow(gt, cmap=viridis)  # cmap='gray')#
            axs[1].set_title('Expert', **hfont, fontsize=fontsize_)
        else:
            axs[1].set_title('No ground truth provided', **hfont, fontsize=fontsize_)
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        axs[1].set_aspect('equal')

        axs[2].imshow(img, cmap='gray')
        if len(prd):
            prd = np.ma.masked_where(prd==0, prd)
            axs[2].imshow(prd,  viridis)
            axs[2].set_title('CNN (Dice score: {dice}%)'.format(dice=dice), **hfont, fontsize=fontsize_)
        else:
            axs[2].set_title('predicted image not found'.format(dice=dice), **hfont, fontsize=fontsize_)
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        axs[2].set_aspect('equal')

        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        # plt.savefig('images/' + str(file_name) + '.png', dpi=300)
        plt.show()


def display_image(im_display: ndarray, identifier: str = None):
    """ display given array of images.

    Args:
        im_display: array of images to show
        identifier: patient name to display as title
    """
    plt.figure(figsize=(10, 1))
    plt.subplots_adjust(hspace=0.015)
    plt.suptitle("Showing image: " + str(identifier), fontsize=12, y=0.95)
    # loop through the length of tickers and keep track of index
    for n, im in enumerate(im_display):
        # add a new subplot iteratively
        plt.subplot(int(len(im_display) // 2), 2, n + 1)
        plt.imshow(np.log(im + 1))
    plt.show()


def read_predicted_images(path: str = None):
    list_input_dir = os.listdir(path)
    print(f'Number of cases: {len(list_input_dir)}')

    for file_name in list_input_dir:
        current_file = os.path.join(path, file_name)
        # read ct, gt, and pet, and pred
        pet_gt_prd = [ntpath.basename(nii) for nii in glob.glob(str(current_file) + "/*.nii")]
        gt, pet, pred = [], [], []
        # try:
        for index in pet_gt_prd:
            if "pet" in str(index).lower():
                pet = np.asanyarray(nib.load(str(current_file) + "/" + str(index)).dataobj)
            elif "predicted" in str(index).lower():
                pred = np.asanyarray(nib.load(str(current_file) + "/" + str(index)).dataobj)
            elif "ground_truth" in str(index).lower() or "gt" in str(index).lower():
                gt = np.asanyarray(nib.load(str(current_file) + "/" + str(index)).dataobj)

        if len(pred):
            pred[pred>0.5] =1
            pred[pred<0.5] = 0

        for coronal_sagittal in range(2):
            if len(gt) and len(pred):
                pet_gt_prd_display = [pet[coronal_sagittal], gt[coronal_sagittal], pred[coronal_sagittal]]
            elif len(pred):
                pet_gt_prd_display = [pet[coronal_sagittal], gt, pred[coronal_sagittal]]
            elif len(gt):
                pet_gt_prd_display = [pet[coronal_sagittal], gt[coronal_sagittal], pred]
            else:
                pet_gt_prd_display = [pet[coronal_sagittal], gt, pred]

            superimpose_segmentation_images(pet_gt_prd_display, file_name=file_name)
        # except:
        #     pass


if __name__ == '__main__':
    # Function to visualize image and clinical data
    print("visualize data")

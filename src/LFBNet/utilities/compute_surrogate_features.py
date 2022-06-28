"""This script computes the surrogate features of the MIP images such as sTMTV and sDmax.
"""
import os
import numpy as np
import pandas as pd
import glob
import csv
import time
from numpy import ndarray
from tqdm import tqdm
from typing import List, Tuple
import random
from scipy.ndimage import label
import nibabel as nib

random.seed(7)


class ComputesTMTVsDmaxFromNii:
    """ computes the surrogate features from given 2D coronal and sagittal masks (ground truths).

    Args:
        data_path: path to the sagittal and coronal masks.
        get_identifier: a label to identify mask or ground truth (gt) from expert and predicted masks (prd).

    Returns:
        Returns a saved csv files with the computed surrogate biomarkers.
    """

    def __init__(self, data_path: str = None, get_identifier: str = "predicted"):

        self.data_path = data_path

        self.get_identifier = get_identifier

    def compute_and_save_surrogate_features(self):
        """ Compute the surrogate features and save as csv file.

        Returns:
            Saved surrogate features from either the expert or from the ai in csv file.

        """
        # get list of data to compute the surrogate features
        case_ids = [index for index in os.listdir(self.data_path) if not str(index).endswith('csv')]

        # voxel size of the images is considered constant
        voxel_size = [4.0, 4.0, 4.0]

        def get_features(mask_to_compute_feature_on: ndarray = None):
            """ Computes the surrogate MTV and surrogate dissemination.
            Args:
                mask_to_compute_feature_on: binary mask to compute the surrogate features.

            Returns:
                Returns the computed surrogate MTV and dissemination along the height (z) and width (xy) of the image.

            """
            prof_xy, prof_z, stmtv = ComputesTMTVsDmaxFromNii.num_white_pixels(mask_to_compute_feature_on.copy())
            sdmax_xy = ComputesTMTVsDmaxFromNii.compute_surrogate_dissemination(prof_xy, percentile=[2, 98])
            sdmax_z = ComputesTMTVsDmaxFromNii.compute_surrogate_dissemination(prof_z, percentile=[2, 98])
            return stmtv, sdmax_xy, sdmax_z

        # store all calculated features:
        case_name_sagittal_coronal_axial_x_y_z_features = [
            ['PID', 'sTMTV_sagittal', 'sTMTV_coronal', "sTMTV_(mm\u00b2)", 'Sagittal_xy', 'Sagittal_z', 'Coronal_xy',
             'Coronal_z', "sDmax_(mm)", "sDmax_(mm)_euclidean", 'X', 'Y', 'Z']]

        for n, id in tqdm(enumerate(case_ids), total=(len(case_ids))):
            img_folder = os.path.join(self.data_path, str(id))
            case_ids_img_name = os.listdir(img_folder)

            # if there is any ids that ends with _0 or _1, the coronal and sagittal images are saved separately,
            # otherwise not. sagittal with '_o' and coronal with '_1'.
            saved_sagittal_coronal_separately = any([True for case_id in case_ids_img_name if "_sagittal" in case_id])

            # dictionary to store the values for sagittal and coronal features
            sagittal = dict(smtv=0, sdmax_xy=0, sdmax_z=0)
            coronal = dict(smtv=0, sdmax_xy=0, sdmax_z=0)

            for index, read_image in tqdm(enumerate(case_ids_img_name), total=(len(case_ids_img_name))):
                # get number of files ending with the identifier, i.e., predicted (prd) or ground truth (gt).
                if str(self.get_identifier) in str(read_image):
                    read_image_path = os.path.join(img_folder, read_image)
                    mask, _ = ComputesTMTVsDmaxFromNii.get_image(read_image_path)

                    if saved_sagittal_coronal_separately:
                        # We have sagittal and coronal images saved separately.
                        if "_sagittal" in str(read_image):  # sagittal
                            sagittal['smtv'], sagittal['sdmax_xy'], sagittal['sdmax_z'] = get_features(mask)
                        elif "_coronal" in str(read_image):  # coronal
                            coronal['smtv'], coronal['sdmax_xy'], coronal['sdmax_z'] = get_features(mask)
                    else:
                        # sagittal and coronal given as one nifti image.
                        for sagittal_coronal in range(2):
                            mask_ = mask[sagittal_coronal]
                            if sagittal_coronal == 0:  # sagittal
                                sagittal['smtv'], sagittal['sdmax_xy'], sagittal['sdmax_z'] = get_features(mask_)
                            else:  # coroanl
                                coronal['smtv'], coronal['sdmax_xy'], coronal['sdmax_z'] = get_features(mask_)

            # combine the sagittal and coronal features, and compute them in physical space.
            sTMTV, sDmax_abs, sDmax_sqrt = ComputesTMTVsDmaxFromNii.compute_features_in_physical_space(
                sagittal, coronal
                )
            # add the given patient's features into all dataset.
            case_name_sagittal_coronal_axial_x_y_z_features.append(
                [str(id), sagittal['smtv'], coronal['smtv'], sTMTV, sagittal['sdmax_xy'], sagittal['sdmax_z'],
                 coronal['sdmax_xy'], coronal['sdmax_z'], sDmax_abs, sDmax_sqrt, voxel_size[0], voxel_size[1],
                 voxel_size[2]]
                )

        # save the computed features into csv file
        ComputesTMTVsDmaxFromNii.write_it_to_csv(
            data=case_name_sagittal_coronal_axial_x_y_z_features, dir_name=self.data_path,
            identifier=self.get_identifier
            )

        return case_name_sagittal_coronal_axial_x_y_z_features

    @staticmethod
    def compute_features_in_physical_space(
            sagittal: dict = None, coronal: dict = None, voxel_size=None
            ):
        """ Compute features in physical space. Basically multiply the given TMV or dissemination by the voxel space.

        Args:
            sagittal: sagittal view features.
            coronal: coronal view features
            voxel_size: voxel size of the mask

        Returns:
            Returns the biomarker features in physical space, e.g., in mm.

        """
        # add the surrogate volume of the coronal and sagittal images.
        if voxel_size is None:
            voxel_size = [4.0, 4.0, 4.0]
        sTMTV = (sagittal['smtv'] + coronal['smtv']) * voxel_size[0] * voxel_size[2]

        # add the dissemination of the coronal and sagittal images. Absolute distance.
        sDmax_abs = sagittal['sdmax_xy'] * voxel_size[0] + sagittal['sdmax_z'] * voxel_size[2] + coronal['sdmax_xy'] * \
                    voxel_size[0] + coronal['sdmax_z'] * voxel_size[2]

        # add the dissemination of the coronal and sagittal images. Square distance.
        sDmax_ecludian = np.sqrt(
            np.power(sagittal['sdmax_xy'] * voxel_size[0], 2) + np.power(
                sagittal['sdmax_z'] * voxel_size[2], 2
                ) + np.power(
                coronal['sdmax_xy'] * voxel_size[0], 2
                ) + np.power(coronal['sdmax_z'] * voxel_size[2], 2)
            )

        return sTMTV, sDmax_abs, sDmax_ecludian

    @staticmethod
    def threshold(input_image: ndarray = None, cut_off: float = 0.5):
        """ threshold the input array value using the input cut-off.

        Args:
            input_image: array data to threshold
            cut_off: thresholding value

        Returns:
            Returns the binary image threshold from the cut-off value array.

        """
        input_image = np.array(input_image)
        input_image[input_image < cut_off] = 0
        input_image[input_image >= cut_off] = 1
        return input_image

    @staticmethod
    def num_white_pixels(input_image):
        """ Computes the surrogate volume of the mask, and the projection profile of the non-zero values into
        horizontal (z)
        and vertical profiles (xy).

        Args:
            Input: binary image mask img.

        Returns:
            Returns the number of pixels across each row of the image in profile_axis_z
        and number of pixels across each column of the image in profile axis_xy.
        """
        input_image = ComputesTMTVsDmaxFromNii.threshold(input_image)

        # remove small nodes < 4.8 cm2
        input_image = ComputesTMTVsDmaxFromNii.remove_outliers(input_image, minimum_cc_sum=30)

        profile_axis_z, profile_axis_xy = [], []
        for index in range(input_image.shape[0]):
            profile_axis_xy.append(np.sum(input_image[index, :] > 0))
        for index in range(input_image.shape[1]):
            profile_axis_z.append(np.sum(input_image[:, index] > 0))

        smtv = np.sum(input_image > 0)

        return profile_axis_xy, profile_axis_z, smtv

    @staticmethod
    def remove_outliers(input_image: ndarray = None, minimum_cc_sum: float = None):
        """ Remove outliers. Outliers are considered to be small regions (less than the given minimum_cc_sum).

        Args:
            input_image: input mask.
            minimum_cc_sum: threshold value to remove tumor size less than this value.

        Returns:
            Returns mask with outliers or small independent regions of tumor removed.

        """
        binary_mask = input_image.copy()
        labelled_mask, num_labels = label(binary_mask)
        # Let us now remove all the small regions, i.e., less than the specified mm.
        refined_mask = input_image.copy()
        for get_label in range(num_labels):
            if np.sum(refined_mask[labelled_mask == get_label]) <= minimum_cc_sum:
                refined_mask[labelled_mask == get_label] = 0
        return refined_mask

    @staticmethod
    def compute_surrogate_dissemination(input_surrogate: List[ndarray] = None, percentile: List = None):
        """ calculate the dissemination.

        Args:
            input_surrogate: input mask image.
            percentile: percentile of the tumor regions to consider for the dissemination computation.

        Returns:
            Returns calculated surrogate tumor dissemination.

        """
        # if the input image has tumor regions, foreground regions.
        if np.asarray(input_surrogate).any() > 0:
            # get the positions where there is tumor region
            ls = [index for index, element in enumerate(input_surrogate) if element != 0]
            # distance is the difference between the first and the last index plus one. s
            n = int(ls[-1] - ls[0] + 1)
            distance = np.absolute(np.ceil(n * percentile[0] / 100) - np.ceil(n * percentile[1] / 100))
        else:
            distance = 0
        return distance

    @staticmethod
    def get_image(img_mask_pt):
        """ get nifti image

        Args:
            img_mask_pt:path to the mask image to read.

        Returns:
            Returns threshold mask image.
        """
        # get the nifti image
        mask = nib.load(img_mask_pt)
        # get the voxel spacing
        voxel_size = mask.header.get_zooms()

        mask = np.asanyarray(mask.dataobj)
        mask = ComputesTMTVsDmaxFromNii.threshold(mask)
        return mask, voxel_size

    @staticmethod
    def write_it_to_csv(data: List = None, dir_name: str = None, identifier: str = "predicted"):
        """ Write the surrogate feature in xls files

        Args:
            data: input array to write
            dir_name: path to save the xls/csv file
            identifier: unique identifier or name of the xls file
        """

        data = np.array(data)
        if dir_name is None:
            dir = "./csv"
        else:
            dir = os.path.dirname(dir_name)

        if not os.path.exists(dir):
            os.mkdir(dir)

        file_name = os.path.join(dir, 'surrogate_' + str(identifier) + '.csv')
        with open(file_name, 'w', newline='') as output:
            output_data = csv.writer(output, delimiter=',')

            # write the column name in the first row
            # columns = [str(i) for i in range(data.shape[1])]
            #
            # output_data.writerow(columns)

            for row_in_data in range(data.shape[0] + 1):
                # first row jump it
                if row_in_data != 0:
                    output_data.writerow(data[row_in_data - 1][:])

        print("Total data processed %0.3d" % len(data))
        print("CSV file saved to: ", dir)


if __name__ == '__main__':
    data_pth = r"E:\ai4elife\data\predicted/"
    cls = ComputesTMTVsDmaxFromNii(data_path=data_pth, get_identifier="predicted")
    cls.compute_and_save_surrogate_features()

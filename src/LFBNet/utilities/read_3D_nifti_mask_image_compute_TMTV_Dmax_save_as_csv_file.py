""" Script to compute metabolic tumor volume (TMTV) and dissemination from a given 3D mask images.
1. read .nii files
2. read the pixel spacing
3. compute the total metabolic tumor volume (TMTV)
4. Compute the lesion dissemination (Dmax)
5. calculate TMTV and Dmax in physical spacing, using the pixel spacing
6. save  in .CSV as columns of: patient name (anonymous), pixel_spaing, TMTV, Dmax
"""

# import important libraries
import os
import glob
from tqdm import tqdm
import argparse
from numpy.random import seed
seed(1)

import nibabel as nib
import numpy as np
import csv

# library for dmax computation
from skimage.measure import label,  regionprops
from skimage import data, util
from scipy.spatial import  distance

import pathlib



# function to write csv file
def write_to_csv_file(array, output_path, file_name="csv"):
    """
    :param array: array that consists rows and columns to be saved to csv file
    :param output_path: The directory to save csv file
    :param file_name: Name of the csv file
    :return: saved file_name.csv in the older output_path
    """
    array = np.array(array)
    file_name = str(output_path) + '/' + str(file_name) + '.csv'

    with open(file_name, 'w', newline='') as output:
        output_data = csv.writer(output, delimiter=",")
        for row in range(array.shape[0]):
            output_data.writerow(array[row][:])

    print("saved at: ", file_name)


# function to read .nii and compute biomarker values
def read_nii_mask_save_csv_tmtv_dmax(input_path, output_path):
    """
    :param input_path: Path to the directory that consists the directory for .nii files
    :param output_path:  The directory to save csv file, after computing the TMTV and Dmax, pixel spacing
    :return: read .nii, compute TMTV, Dmax
    """
    case_ids = os.listdir(input_path)
    print("Total number of cases to read: 0.1%d", len(case_ids))
    name_x_y_z_TMTV_dmax = [["ID", "X", "Y", "Z", 'TMTV', "Dmax"]]


    for n, case_name in tqdm(enumerate(case_ids), total=len(case_ids)):
        path_img_nii = str(input_path) + "/" + str(case_name) + "/gt/"
        path_img_nii = glob.glob(path_img_nii + "/*.nii.gz")[0]
        try:
            # Read .nii files
            gt = nib.load(path_img_nii)
            res_pet = gt.header.get_zooms()
            gt = np.asanyarray(gt.dataobj)
            gt[gt > 0] = 1
            gt[gt <= 0] = 0

            # Compute TMTV
            def compute_TMTV(gt_):
                gt_[gt_ > 0] = 1
                gt_[gt_ <= 0] = 0
                return np.sum(gt_ > 0)

            #compute dmax
            def compute_dmax(gt_):
                # label images
                gt_ = util.img_as_ubyte(gt_) > 0
                gt_= label(gt_, connectivity=gt_.ndim)
                props = regionprops(gt_)

                dist_all = []
                for k in range(len(props)):
                    # physical space
                    a = np.multiply(np.array(props[k].centroid), np.array(res_pet))

                    # compute the distance between all other centroids:
                    if len(props) >1:
                        for kk in range(len(props)):
                            b = np.multiply(np.array(props[kk].centroid), np.array(res_pet))
                            dist = distance.euclidean(a, b)
                            dist_all.append(dist)
                    else:
                        dist_all.append(0)
                return np.max(dist_all)

            tmtv = compute_TMTV(gt.copy())
            dmax = compute_dmax(gt.copy())
            name_x_y_z_TMTV_dmax.append([str(case_name), res_pet[0], res_pet[1], res_pet[2],
                                         tmtv * res_pet[0] * res_pet[1] * res_pet[2], dmax])
        except:
            print(f"Error reading {path_img_nii}")
            continue

    write_to_csv_file(name_x_y_z_TMTV_dmax, output_path, file_name="data_xyz_tmtv_dmax")
    print('Total number of patients correctly read and their volume calculated: ', len(name_x_y_z_TMTV_dmax) - 1)
    print("Done !!")


if __name__ == "__main__":
    # We assume the .nii file name and the folder name are the same
    parser = argparse.ArgumentParser(description="script to read nii files and compute TMTV and Dmax")
    parser.add_argument("--input_dir", dest='input_dir',  type=pathlib.Path, help="Input directory path to .nii files")
    parser.add_argument("--output_dir", dest='output_dir', type=pathlib.Path, help='output directory path')
    args = parser.parse_args()
    read_nii_mask_save_csv_tmtv_dmax(args.input_dir, args.output_dir)

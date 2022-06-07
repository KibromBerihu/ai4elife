"""script to predict the segmentation results and calculate surrogate biomarkers of a given testing dataset using the docker image.

This script allows users to execute the whole pipeline using the docker image.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from src.LFBNet.utilities import train_valid_paths
from src.LFBNet.preprocessing import preprocessing
from src.run import trainer, parse_argument
from src.LFBNet.utilities.compute_surrogate_features import ComputesTMTVsDmaxFromNii


def main():
    """ Predicts tumor segmentation results and calculates associated quantitative metrics on a given testing dataset.



    This function receives the path directory to the testing dataset that contains the PET images. It predicts the

    segmentation results and saves them as .nii files. It then calculates the surrogate metabolic tumor volume (sTMTV) and

    surrogate dissemination feature (sDmax) and saves it as CSV or Xls file.

    Acronyms:
        PET: Nifti format of [18]F-FDG PET images in SUV unit.
        GT: Ground truth mask from the expert if available.

    [directory_path_to_raw 3D nifti data with SUV values] should have the following structure as:
    main_dir:
        -- patient_id_1:
            -- PET
                --give_name.nii or give_name.nii.gz
            -- GT (if available) (Ground truth mask from the expert if available)
                -- give_name.nii or give_name.nii.gz

         -- patient_id_2:
            -- PET
                --give_name.nii or give_name.nii.gz
            -- GT (if available)
                -- give_name.nii or give_name.nii.gz

    It reads the .nii files, resize, crop, and save the 3D data, then from these data it generates the sagittal and
    coronal PET MIPs and the ground truth (mask from the expert) if available in the folder.

    Get the latest trained model weight from './weight' directory and use that weight to predict the segmentation.

    Returns:
        save segmented images and computed surrogate biomarker features using the last weight saved in the ./weight
        folder.
    """

    # Path to the parent/main directory. Please read readme.md for how to organize your files.
    input_dir = "/input"

    # parameters to set
    dataset_name = 'data'
    desired_spacing = [4.0, 4.0, 4.0]

    # path to the preprocessed data
    preprocessing_data_dir = "/output"

    preprocessing_params = dict(
        data_path=input_dir, data_name=dataset_name, saving_dir=preprocessing_data_dir, save_3D=True,
        output_resolution=[128, 128, 256], desired_spacing=desired_spacing, generate_mip=True
        )
    mip_data_dir = preprocessing.read_pet_gt_resize_crop_save_as_3d_andor_mip(**preprocessing_params)

    # get list of all patient names from the generated mip directory
    patient_id_list = os.listdir(mip_data_dir)
    print('There are %d cases to evaluate \n' % len(patient_id_list))

    # prediction on the given testing dataset
    test_params = dict(
        preprocessed_dir=mip_data_dir, data_list=patient_id_list, predicted_dir=preprocessing_data_dir
        )
    network_run = trainer.ModelTesting(**test_params)
    network_run.test()

    print("\n\n Computing the surrogate biomarkers ... \n\n")
    for identifier, data_path in zip(
            ["predicted", "ground_truth"], [os.path.join(preprocessing_data_dir, "predicted_data"),
                            os.path.join(preprocessing_data_dir, "data_default_MIP_dir")]
            ):
        try:
            csv_file = ComputesTMTVsDmaxFromNii(data_path=data_path, get_identifier=identifier)
            csv_file.compute_and_save_surrogate_features()
        except:
            continue


# check
if __name__ == '__main__':
    print("\n Running the integrated framework for testing use case... \n\n")
    main()

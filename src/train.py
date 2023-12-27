""" training, transfer learning, and validation (testing with reference ground truth images) of the proposed deep
learning model.

The script allows to train a deep learning model on a given PET images in folder with corresponding ground truth (GT).
It is assumed that the directory structure of the dataset for training and validation are given as follows:
    main_dir:
        -- patient_id_1:
            -- PET
                --give_name.nii [.gz]
            -- GT
                -- give_name.nii [.gz]

         -- patient_id_2:
            -- PET
                --give_name.nii [.gz]
            -- GT
                -- give_name.nii [.gz]

Please refer to the requirements.yml or requirements.txt files for the required packages to run this script. Using
anaconda virtual environment is recommended to runt the script.

e.g. python train.py  --input_dir path/to/input/data --task [train or valid]
python train.py --input_dir ../data/hecktor_nii_cropped/ --task train

By K.B. Girum
"""
import os
import sys

# setup directory
p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

from LFBNet.utilities import train_valid_paths
from LFBNet.preprocessing import preprocessing
from run import trainer, parse_argument


def main():
    """ train and/or validate the selected model configuration. Get parsed arguments from parse_argument function, and
    preprocess data, generate MIP, and train and validate.

     :parameter:
           --input_dir [path_to_pet_images]
           -- dataset_name [unique_dataset_name]
           -- output_dir [path_to_save_predicted_values] [optional]
           -- task test # testing model of the model [Optional]

     :returns:
            - trained model if the task is train, and the predicted segmentation results if the task is valid.
            - predicted results will be saved to predicted folder when the task is to predict.
            - It also saves the dice, sen, and specificity of the model on each dataset and the average and median
            values.
            - It computes the quantitative surrogate metabolic tumor volume (sTMTV) and surrogate dissemination feature
            (sDmax) from the segmented and ground truth images and saves them as xls file. The xls file column would
            have
            [patient_id, sTMTV_gt, Sdmax_gt, TMTV_prd, and Dmax_prd].
            - Note pred: predicted estimate, and gt: ground truth or from expert.

    """
    # get the parsed arguments, such as input directory path, output directory path, task, test or training
    args = parse_argument.get_parsed_arguments()

    # get input and output data directories
    input_dir = args.input_dir
    train_valid_paths.directory_exist(input_dir)  # CHECK: check if the input directory has files

    # data identifier, or name
    dataset_name = args.data_identifier

    # how to split the training and validation data:
    # OPTION 1: provide a csv file with two columns of list of patient ids: columns 1 ['train'] and column 2 ['valid'].
    # [set args.from_csv = True].
    # OPTION 2: let the program divide the given whole data set into training and validation data randomly.
    # [set args.from_csv = False].

    train_valid_id_from_csv = args.from_csv

    # output directory to save
    if args.output_dir:  # if given
        output_dir = args.output_dir
    else:
        # if not given it will create under the folder "../../data/  str(dataset_name) + 'default_3d_dir'
        output_dir = '../data/predicted'  # directory to the MIP
        if not os.path.exists('../data'):
            os.mkdir('../data')

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # processed directory:
    preprocessing_dir = '../data/preprocessed'  # directory to the MIP
    if not os.path.exists('../data'):
        os.mkdir('../data')

    if not os.path.exists(preprocessing_dir):
        os.mkdir(preprocessing_dir)

    # default output data spacing
    desired_spacing = [4.0, 4.0, 4.0]

    # STEP 1:  read the raw .nii files in suv and resize, crop in 3D form, generate MIP, and save
    # get the directory path to the generated and saved MIPS, if it already exists, go for training or testing
    dir_mip = []
    # path to the training and validation data
    path_train_valid = dict(train=None, test=None)

    # preprocessing stage:
    preprocessing_params = dict(data_path=input_dir, data_name=dataset_name, saving_dir=preprocessing_dir, save_3D=True,
    output_resolution=[128, 128, 256], desired_spacing=desired_spacing, generate_mip=True)

    dir_mip = preprocessing.read_pet_gt_resize_crop_save_as_3d_andor_mip(**preprocessing_params)

    # training or validation/testing from the input argument task
    task = args.task  # true training and false testing or validation

    # training deep learning model
    if task == 'train':
        # get valid id from the csv file: rom escel files manually set: assuming this csv file has two column,
        # with 'train' column for training data
        if train_valid_id_from_csv:
            train_valid_ids_path_csv = r'../csv/'
            train_ids, valid_ids = trainer.get_training_and_validation_ids_from_csv(train_valid_ids_path_csv)
        else:
            # generate csv file for the validation and training data by dividing the data into training and validation
            path_train_valid = dict(train=dir_mip)
            train_ids, valid_ids = train_valid_paths.get_train_valid_ids_from_folder(path_train_valid=path_train_valid)

        # train or test on the given input arguments
        trainer_params = dict(folder_preprocessed_train=dir_mip, folder_preprocessed_valid=dir_mip,
                              ids_to_read_train=train_ids, ids_to_read_valid=valid_ids, task=task,
                              predicted_directory=output_dir)

        network_run = trainer.NetworkTrainer(**trainer_params)
        network_run.train()

    # validation
    elif task == 'valid':
        dir_mip = os.path.join(preprocessing_dir, str(dataset_name) + "_default_MIP_dir")

        # get valid id from the csv file: assume training ids are under column name "train" and testing under "test"
        if train_valid_id_from_csv:
            train_valid_ids_path_csv = r'../csv/'
            train_ids, valid_ids = trainer.get_training_and_validation_ids_from_csv(train_valid_ids_path_csv)

        else:
            # generate csv file for the validation and training data by dividing the data into training and validation
            path_train_valid = dict(train=dir_mip)
            train_ids, valid_ids = train_valid_paths.get_train_valid_ids_from_folder(path_train_valid=path_train_valid)

        trainer_params = dict(folder_preprocessed_train=dir_mip, folder_preprocessed_valid=dir_mip,
                              ids_to_read_train=train_ids, ids_to_read_valid=valid_ids, task=task,
                              predicted_directory=output_dir, save_predicted=True)

        network_run = trainer.NetworkTrainer(**trainer_params)
        network_run.train()

    else:
        print("key word %s not recognized !\n" % task)


# check
if __name__ == '__main__':
    print("Running the integrated framework ... \n\n")
    main()

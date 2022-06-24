"""
Copyright 2022 LITO, Institut Curie, Paris, France
write the license issue

"""

import os
import numpy as np
import pandas as pd
import glob
import csv
import time

import random

random.seed(7)


def write_it_to_csv(data, name='file', dir_name=None, columns_=None):
    """

    Args:
        data:
        name:
        dir_name:
        columns_:
    """
    data = np.array(data)
    if dir_name is None:
        dir = "../csv"
    else:
        dir = "../" + str(dir_name) + '_csv'

    if not os.path.exists(dir):
        os.mkdir(dir)

    file_name = str(dir) + '/' + str(name) + "_" + str(time.time()) + '.csv'
    with open(file_name, 'w', newline='') as output:
        output_data = csv.writer(output, delimiter=',')

        # write the column name in the first row
        if columns_ is None:
            columns = ['column_' + str(i) for i in range(data.shape[1])]
        else:
            columns = [clm for clm in columns_]

            if len(columns) < data.shape[1]:
                add_column_name = ['column_' + str(i) for i in range(abs(data.shape[1] - len(columns)))]
                columns.extend(add_column_name)

        output_data.writerow(columns)

        for row_in_data in range(data.shape[0] + 1):
            # first row jump it
            if row_in_data != 0:
                output_data.writerow(data[row_in_data - 1][:])
    print("CSV file saved to: ", os.getcwd())


def directory_exist(dir_check: str = None):
    """
    :param dir_check:
    """
    if os.path.exists(dir_check):
        print("The directory %s does exist \n" % dir_check)
        pass
    else:
        raise Exception("Please provide the correct path to the processed data: %s not found \n" % (dir_check))


def read_id_from_dir(path_dir: str = None):
    """

    :param path_dir:
    :return:
    """
    case_ids = []
    for id in os.listdir(path_dir):
        case_ids.append(id)
    # make permutation in the given list
    case_ids = np.array(case_ids)
    # indices = np.random.permutation(len(case_ids))
    # case_ids = case_ids[indices]
    return case_ids, case_ids


def get_training_and_validation_ids_from_csv(path):
    """
    Considers the training and validation data are under one folder, and train and valid ids are given in csv file as
    train, and valid columns
    :param path:
    :return:
    """
    ids = read_csv_train_valid_index(path)
    train_ids = ids[0]
    valid_ids = ids[1]

    return train_ids, valid_ids


def get_train_valid_ids_from_folder(path_train_valid: dict = None, ratio_valid_data: int = 0.25):
    """
    gets the path to the train and validation data as dictionary, with dictionary name "train" and "valid".
    if only the train or valid is given it considers random separation of training and validation ids with
    ratio_valid_data.
    The default value is 25%

    :param path_train_valid: dictionary of path to training data, with key word "train"
    :param ratio_valid_data: dictionary of path to validation data, with key word "valid"
    :return:
            -- save valid and train ids in the current directory of under sub folder .csv
            -- return trained and validation ids
    """
    # given training and validation data on one folder, random splitting with .ratio_valid_data% : train, valid
    if len(path_train_valid.keys()) == 1:
        all_cases_ids = os.listdir(str(path_train_valid['train']))  # all patients id

        # make permutation in the given list
        all_cases_ids = np.array(all_cases_ids)
        indices = np.random.permutation(len(all_cases_ids))
        num_valid_data = int(round(ratio_valid_data * len(all_cases_ids)))

        train_ids = indices[num_valid_data:]
        valid_ids = indices[:num_valid_data]

        train_id, valid_id = all_cases_ids[train_ids], all_cases_ids[valid_ids]

    else:  # if the training and validation paths are given individually as array [train, valid]
        # read all directories of training and validation

        train_id = os.listdir(path_train_valid['train'])
        valid_id = os.listdir(path_train_valid['valid'])

    # Check for data leakage between the training and validation ids:
    print("Number of training: %d  and validation: %d \n" % (len(train_id), len(valid_id)))
    leaked_cases_train_valid = [leaked_cases for leaked_cases in list(train_id) if leaked_cases in list(valid_id)]
    if len(leaked_cases_train_valid):
        raise Exception(
            "Some cases are redundant in the training and validation data, "
            "e.g. %s" % leaked_cases_train_valid
            )

    # combine the training and testing names into arrays
    # write to unequal array into one array and save them into csv
    train_test = []
    for i in range(len(valid_id)) if len(valid_id) > len(train_id) else range(len(train_id)):
        try:
            train_test.append((train_id[i], valid_id[i]))
        except:
            try:
                train_test.append((train_id[i], None))
            except:
                train_test.append((None, valid_id[i]))
    # save it for later reference

    print(train_test)
    write_it_to_csv(
        train_test, name='train_valid_ids', columns_=['train', 'valid']
        )
    return train_id, valid_id


def get_output_or_create_folder_name(
        model: str, task: str = None, trainer: str = None, pans: str = None, fold: int = None,
        processed_data_directory: str = None
        ):
    """
    Retrieve the output directory for the LFB-net model given in the input parameters
    :param processed_data_directory:
    :param model:
    :param task:
    :param trainer:
    :param pans:
    :param fold:
    :return:
    """

    assert model in ['2D', "3D"]

    # check if the directory to the processed data is given
    directory_exist(processed_data_directory)
    # if it exists
    data_dir = processed_data_directory

    # training data directory, and validation data directory
    training_dir = "/".join(data_dir, "train")
    valid_dir = "/".join(data_dir, "valid")

    # check if the directory for the training and validation data exists
    directory_exist(training_dir)
    directory_exist(valid_dir)

    # Get list of the training and validation ids
    train_id = read_id_from_dir(training_dir.copy())
    valid_id = read_id_from_dir(valid_dir.copy())

    # Check for data leakage between the training and validation ids:
    print("Number of training: %d  and validation: %d \n" % (len(train_id), len(valid_id)))
    leaked_cases_train_valid = [leaked_cases for leaked_cases in list(train_id) if leaked_cases in list(valid_id)]
    if len(leaked_cases_train_valid):
        raise Exception(
            "Some cases are redundant in the training and validation data, "
            "e.g. %s" % leaked_cases_train_valid
            )

    return [training_dir, valid_dir, train_id, valid_id]


def read_Excel_or_Csv_file(path_here):
    """

    :param path:
    :return:
    """
    try:
        excelData = pd.read_excel(str(path_here))
    except:
        excelData = pd.read_csv(str(path_here))
    return excelData


# given the path of a folder consistes of train and validation, and train, valid ids in csv
def read_csv_train_valid_index(path_, csv_identifier: str = None):
    """

    :param csv_identifier:
    :param path_:
    : csv_identifier: name of the csv file to read
    :return:
    """
    # get all csv files in the provided path
    types = ('/*.csv', '/*.xls')  # the tuple of file types
    csv_xls_files = []
    for files in types:
        csv_xls_files.extend([i for i in glob.glob(str(path_) + files)])

    train_id, valid_id = [], []
    if csv_identifier is not None:
        for csv_file in csv_xls_files:
            identifier_base_name = str(os.path.basename(csv_file)).split('.')[0]
            # csv to select
            if str(identifier_base_name) == str(csv_identifier):  # select to provide csv file name
                # setting first name as index column
                excel_data = read_Excel_or_Csv_file(csv_file)
                if bool(len(excel_data.columns)):
                    train_id = excel_data["train"].dropna()
                    valid_id = excel_data["valid"].dropna()
                else:
                    pass

    else:  # if the csv_identifer is not given, take the first csv file
        # setting first name as index column
        csv_read = csv_xls_files[0]
        excel_data = read_Excel_or_Csv_file(csv_read)
        if bool(len(excel_data.columns)):
            train_id = excel_data["train"].dropna()
            valid_id = excel_data["valid"].dropna()
        else:
            pass
            raise ValueError("Excel or CSV file not found")

    # Check for data leakage between the training and validation ids:
    print("Number of training: %d  and validation: %d \n" % (len(train_id), len(valid_id)))
    leaked_cases_train_valid = [leaked_cases for leaked_cases in list(train_id) if leaked_cases in list(valid_id)]
    if len(leaked_cases_train_valid):
        raise Exception(
            "Some cases are redundant in the training and validation data, "
            "e.g. %s" % leaked_cases_train_valid
            )

    return [train_id, valid_id]


if __name__ == '__main__':
    # print("traind_valid_path_finder ")
    path_ = r"F:\LFB_Net\data\csv\training_validation_indexs\remarc/"
    xy = read_csv_train_valid_index(path_)

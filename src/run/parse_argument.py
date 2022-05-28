""" Parse input arguments
By K.B. Girum
"""
import argparse
import pathlib


def get_parsed_arguments():
    """ parses the following important inputs arguments.

    :parameters:
        --input_dir: path to the raw pet and ground truth (gt)  folders
        --output_dir: path to save the preprocessed and predicted images (you can ignore it)
        --data_id: unique id for the given dataset
        --task: task to perform by the function such as training, validation, testing

    :returns: input directory path, output directory path, preprocessed 3D images saved or not
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', dest='input_dir', type=pathlib.Path, required=True, help='input directory '
                                                                                'path'
        )

    parser.add_argument(
        '--from_csv', default=False, type=bool,
        help='set true if you provide a csv file with training and validation ids'
             'otherwise set false.'
        )

    parser.add_argument('--output_dir', dest='output_dir', type=pathlib.Path, help='output directory path')
    parser.add_argument(
        '--data_id', dest='data_identifier', type=str, help='Unique data Name/identifier', required=True
        )
    parser.add_argument('--task', dest='task', choices=['train', 'valid'], help='set training or validataion mode')
    args = parser.parse_args()

    return args


def get_parsed_arguments_test_case():
    """ parses the following important inputs arguments.

    Args:
         --input_dir: path to the raw pet and ground truth (gt)  folders (you can ignore it).
        --output_dir: path to save the preprocessed and predicted images (you can ignore it).
    Returns:
        Returns the path to the input data and output data for easy use casee or testing case.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', dest='input_dir', default=None, type=pathlib.Path, help='path to raw PET images')
    parser.add_argument('--output_dir', dest='output_dir', default=None, type=pathlib.Path, help='output directory path')
    args = parser.parse_args()
    return args


# Check
if __name__ == '__main__':
    print("Get parsed arguments: including input and output directory path\n")
    args_ = get_parsed_arguments()
    print(args_)

"""

"""
import os
import glob
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import nibabel as nib
from typing import List, Tuple
from numpy.random import seed

# seed random number generator
seed(1)


class DataLoader:
    """
    read preprocessed pet and gt MIP data for training
    """

    def __init__(self, data_dir: str, ids_to_read: ndarray = None, shuffle=True, training: bool = True):
        self.data_dir = data_dir
        self.ids_to_read = ids_to_read
        self.shuffle = shuffle
        self.training = training

    def get_batch_of_data(self):
        """
        data structure:
        -- main directory
        ------case Name:
                -- pet.nii.gz
                -- gt.nii.gz
        --Given list of training and testing on .text files
            -- train.text
            -- valid.text
        """

        # check directory
        self.directory_exist(self.data_dir)

        # get all names of the directories under data_dir
        case_ids = os.listdir(self.data_dir)

        # store batch data
        image_batch, ground_truth_batch = [], []

        # if there are file in data dir
        if not len(case_ids):
            raise Exception("No files found in %s" % self.data_dir)

        # else continue getting.reading the files
        for get_id in list(case_ids):
            if str(get_id) in list(self.ids_to_read):
                try:
                    # consider there four images in each folder name get_id:
                    # e.g. : coronal (gt_1, pet_1) and sagittal  (gt_0, pet_0)
                    current_dir = os.path.join(self.data_dir, str(get_id))
                    # read sagittal and coronal as independent images
                    pet_sagittla_coronal, gt_sagittal_coronal = self.get_nii_files_path(current_dir)

                    # pet, normalization, standardization
                    if len(pet_sagittla_coronal):  # if image is read
                        pet_sagittla_coronal = self.data_normalization_standardization(pet_sagittla_coronal,
                                                                                       z_score=True,
                                                                                       z_score_include_zeros=False)

                        gt_sagittal_coronal = self.data_normalization_standardization(gt_sagittal_coronal, threshold=True)

                        # display or save samples
                        # self.mip_show(pet=pet_sagittla_coronal, gt=gt_sagittal_coronal, identifier=str(get_id))

                        # collect all images with case_id
                        if not bool(len(image_batch)):  # if it is empty; first time
                            image_batch = pet_sagittla_coronal
                            ground_truth_batch = gt_sagittal_coronal
                        else:
                            image_batch = np.concatenate((image_batch, pet_sagittla_coronal), axis=0)
                            ground_truth_batch = np.concatenate((ground_truth_batch, gt_sagittal_coronal), axis=0)
                except:
                    print('Not read %s' %(str(get_id)))

        return [image_batch, ground_truth_batch]

    @staticmethod
    def directory_exist(dir_check: str = None) -> None:
        """
        :param dir_check:
        """
        if os.path.exists(dir_check):
            #  print("The directory %s does exist \n" % dir_check)
            pass
        else:
            raise Exception(
                "Please provide the correct path to the processed data ! \n %s not found \n" % (dir_check))

    @staticmethod
    def mip_show(pet: ndarray = None, gt: ndarray = None, identifier: str = None) -> None:
        """

        :param pet:
        :param gt:
        :param identifier:
        :return:
        """
        # consider axis 0 for sagittal and axis 1 for coronal views
        fig, axs = plt.subplots(1, 4, figsize=(15, 15))
        plt.title(str(identifier))
        try:
            pet = np.squeeze(pet)
            gt = np.squeeze(gt)
        except:
            pass

        axs[0].imshow(np.rot90(np.log(pet[0] + 1)))
        axs[0].set_title('pet_project_on_axis_0')
        axs[1].imshow(np.rot90(np.log(gt[0] + 1)))
        axs[1].set_title('gt_project_on_axis_0')
        axs[2].imshow(np.rot90(np.log(pet[1] + 1)))
        axs[2].set_title('project_on_axis_1')
        axs[3].imshow(np.rot90(np.log(gt[1] + 1)))
        axs[3].set_title('gt_project_on_axis_1')
        plt.show()

    @staticmethod
    def get_nii_files_path(data_directory: str) -> List[ndarray]:
        """
        read .nii or .nii.gz files from a given folder of path data_directory
        :param data_directory:
        :return:
        """
        # more than one .nii or .nii.gz is found in the folder the first will be returned
        types = ('/*.nii', '/*.nii.gz')  # the tuple of file types
        nii_paths = []
        for files in types:
            nii_paths.extend([i for i in glob.glob(str(data_directory) + files)])

        pet, gt = [], []
        if not len(nii_paths):  # if no file exists that ends wtih .nii.gz or .nii
            # raise Exception("No .nii or .nii.gz found in %s dirctory" % data_directory)
            pass
        else:
            # assuming the folder contains coronal mips: pet_1, gt_1, and sagittal mips: pet_0, gt_0,
            pet_saggital, pet_coronal, gt_saggital, gt_coronal = [], [], [], []
            for path in list(nii_paths):
                # get the base name: means the file name
                identifier_base_name = str(os.path.basename(path)).split('.')[0]
                if "pet_sagittal" == str(identifier_base_name):
                    pet_saggital = np.asanyarray(nib.load(path).dataobj)
                    pet_saggital = np.expand_dims(pet_saggital, axis=0)

                elif "pet_coronal" == str(identifier_base_name):
                    pet_coronal = np.asanyarray(nib.load(path).dataobj)
                    pet_coronal = np.expand_dims(pet_coronal, axis=0)

                if "ground_truth_sagittal" == str(identifier_base_name):
                    gt_saggital = np.asanyarray(nib.load(path).dataobj)
                    gt_saggital = np.expand_dims(gt_saggital, axis=0)

                elif "ground_truth_coronal" == str(identifier_base_name):
                    gt_coronal = np.asanyarray(nib.load(path).dataobj)
                    gt_coronal = np.expand_dims(gt_coronal, axis=0)

            # concatenate coronal and sagita images
            # show
            pet = np.concatenate((pet_saggital, pet_coronal), axis=0)
            gt = np.concatenate((gt_saggital, gt_coronal), axis=0)
        return [pet, gt]

    @staticmethod
    def z_score(image: ndarray, include_zeros: bool = False):
        """

        :param image:
        :param include_zeros:
        :return:
        """
        # include zeros
        if include_zeros:
            image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        else:
            # Don't include zeros
            means = np.true_divide(image.sum(), (image != 0).sum())
            stds = np.nanstd(np.where(np.isclose(image, 0), np.nan, image))
            image = (image - means) / (stds + 1e-8)
        return image

    def data_normalization_standardization(self, data: ndarray, threshold: bool = False, z_score: bool = False,
                                           z_score_include_zeros: bool = False,
                                           min_max_scale: bool = False, log_transform: bool = False) -> List[ndarray]:
        """
        Data normalization and standardization function
        :param data:
        :param threshold:
        :param z_score:
        :param z_score_include_zeros:
        :param min_max_scale:
        :param log_transform:
        :return:
        """

        if not isinstance(data, List):
            data = np.array(data)

        # groundtruh > 0 is 1 and <=0 is 0
        if threshold:
            data[data > 0] = 1

        if z_score:
            data = self.z_score(data, include_zeros=z_score_include_zeros)

        if min_max_scale:
            data = (data - min(data)) / (max(data) - min(data))

        if log_transform:
            data = np.log(data + 1)

        return data


if __name__ == '__main__':
    # for Example
    print("data_loader for preprocessed coronal and sagittal MIPs, pet, and gt")
    data_dir = "../data/vienna_default_MIP_dir/"
    ids_to_read = os.listdir(data_dir)

    data_loader = DataLoader(data_dir=data_dir, ids_to_read=ids_to_read)
    loaded_data = data_loader.get_batch_of_data()
    print(np.array(loaded_data).shape)

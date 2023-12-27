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

# seed random number generator
seed(1)


def directory_exist(dir_check: str = None):
    """ Check if the given path exists.

    Args:
        dir_check: directory path to check
    Raises:
        Raises exception if the given directory path doesn't exist.
    """
    if os.path.exists(dir_check):
        # print("The directory %s does exist \n" % dir_check)
        pass
    else:
        raise Exception("Please provide the correct path to the processed data: %s not found \n" % (dir_check))


def generate_mip_show(pet: int = None, gt: int = None, identifier: str = None):
    """ Display the MIP images of a given 3D FDG PET image , and corresponding ground truth gt.

    Args:
        pet: 3D array of PET images
        gt: 3D array of reference or ground truth lymphoma segmentation.
        identifier: Name (identifier) of the patient.
    """
    # consider axis 0 for sagittal and axis 1 for coronal views
    pet, gt = np.array(pet), np.array(gt)
    pet = [np.amax(pet, axis=0), np.amax(pet, axis=1)]
    gt = [np.amax(gt, axis=0), np.amax(gt, axis=1)]
    try:
        pet = np.squeeze(pet)
        gt = np.squeeze(gt)
    except:
        pass

    img_gt = np.concatenate((pet, gt, pet), axis=0)
    display_image(img_gt, identifier=identifier)


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


def get_pet_gt(current_dir: str = None):
    """ Read pet and corresponding reference images.

    Args:
        current_dir: directory path to the PET and corresponding ground truth image.
    Returns:
        Returns array of pet and ground truth images.

    """

    def get_nii_files(nii_path):
        """ Read .nii data in the given nii_path.
        if there are more .nii or .nii.gz file inside the nii_path directory, the function will return the
        first read image.

        Args:
            nii_path: directory path to the .nii or .nii.gz file.

        Returns:
            Returns loaded nibble format if it exists, else it returns empty array.

        """
        # more than one .nii or .nii.gz is found in the folder the first will be returned
        full_path = [path_i for path_i in glob.glob(str(nii_path) + "/*.nii.gz")]

        if not len(full_path):  # if no file exists that ends wtih .nii.gz
            full_path = [path_i for path_i in glob.glob(str(nii_path) + "/*.nii")]

        if not len(full_path):
            get_image = None  # raise Exception("No .nii or .nii.gz found in %s dirctory" % nii_path)

        else:
            print("%d files found in %s \t the first will be read " % (len(full_path), nii_path))
            print("reading ... %s" % full_path[0])
            get_image = nib.load(full_path[0])

        return get_image

    # all folders files
    # all_ = [path_i for path_i in glob.glob(str(nii_path) + "/*.nii")]

    # Get pet image
    # if "pet" in str()
    pet_path = str(current_dir) + "/pet/"
    pet = get_nii_files(pet_path)

    # Get gt image
    try:
        # if the ground truth exists
        gt_path = str(current_dir) + "/gt/"
        gt = get_nii_files(gt_path)
    except:
        #  if the ground truth does not exist
        gt = None

    return [pet, gt]


def resize_nii_to_desired_spacing(
        data: int = None, data_spacing: Tuple[float] = None, desired_spacing: ndarray = None,
        interpolation_order_value: int = None
        ):
    """ resizes a given input data into the desired spacing using the specified interpolation order.

    Args:
        data: array of input data to resize.
        data_spacing:  original input data spacing.
        desired_spacing: required output data spacing.
        interpolation_order_value: interpolation order. E.g., 0 for binary images, and 3 for cubic interpolation.

    Returns:
        resized data.

    """
    if desired_spacing is None:
        desired_spacing_x, desired_spacing_y, desired_spacing_z = [4.0, 4.0, 4.0]
    else:
        desired_spacing_x, desired_spacing_y, desired_spacing_z = desired_spacing

    print("Given data spacing \t Desired spacing \n")
    print(data_spacing, "\t", end="")
    print(desired_spacing, "\n")

    if not isinstance(data, list):
        data = np.array(data)

    # New resolution, consider at least the input image is two-dimensional
    new_x_resolution = np.ceil(data.shape[0] * (data_spacing[0] / desired_spacing_x))
    new_y_resolution = np.ceil(data.shape[1] * (data_spacing[1] / desired_spacing_y))

    print('resizing')
    print(np.array(data).shape)

    if len(data_spacing) == 3 and len(np.squeeze(data).shape) == 3:  # 3D input image
        new_z_resolution = np.ceil(data.shape[2] * (data_spacing[2] / desired_spacing_z))

        # resize to new image resolution
        image_resized = resize(
            data, (new_x_resolution, new_y_resolution, new_z_resolution), order=interpolation_order_value,
            preserve_range=True, anti_aliasing=False
            )

    else:  # if the given input image is 2D
        image_resized = resize(
            data, (new_x_resolution, new_y_resolution), order=interpolation_order_value, preserve_range=True,
            anti_aliasing=False
            )

    return image_resized


def z_score(image):
    """ z-score operation on the input data.

    Args:
        image: input data to apply the z-score.

    Returns:
        Standardized value of the input using z-score. Z-score = (input - mean(input))/(std(input))

    """
    image = image.copy()
    if not isinstance(image, ndarray):
        image = np.array(image)

    image_nan = np.where(image > 0, image, np.nan)
    means = np.nanmean(image_nan, axis=(image.shape))
    stds = np.nanstd(image_nan, axis=(image.shape))
    image_norm = (image - means) / (stds + 1e-8)

    return image_norm


# cropping from the center based:
def crop_nii_to_desired_resolution(data: ndarray = None, cropped_resolution: List[int] = None):
    """ Crops the input data in the given cropped resolution.

    Args:
        data: input data to be cropped.
        cropped_resolution: desired output resolution.

    Returns:
        Returns cropped data of size cropped resolution.

    """
    #
    if not isinstance(data, list):  # if data is not array change it to array
        data = np.array(data)

    try:
        data = np.squeeze(data)
    except:
        pass

    if cropped_resolution is not None:
        cropped_resolution = [128, 128, 256]

    print("\n Initial data size \t Cropped data size ")
    print(data.shape, "\t", end=" ")
    print(cropped_resolution, "\n")

    if len(cropped_resolution) == 3 and len(data.shape) == 3:  # 3D data
        x, y, z = data.shape
    else:
        raise Exception("Input image not !")

    # start cropping : get middle x, y, z values by dividing by 2 and subtract the desired center of image resolution
    start_cropping_at_x = (x // 2 - (cropped_resolution[0] // 2))
    start_cropping_at_y = (y // 2 - (cropped_resolution[1] // 2))
    start_cropping_at_z = (z // 2 - (cropped_resolution[2] // 2))

    # check for off sets: mean the new cropping resolution is bigger than the input image's resolution
    off_set_x, off_set_y, off_set_z = 0, 0, 0
    if start_cropping_at_x < 0:
        off_set_x = np.abs(start_cropping_at_x)
        # set the first pixel at zero
        start_cropping_at_x = 0
    if start_cropping_at_y < 0:
        off_set_y = np.abs(start_cropping_at_y)
        # set the first pixel at zero
        start_cropping_at_y = 0
    if start_cropping_at_z < 0:
        off_set_z = np.abs(start_cropping_at_z)
        # set the first pixel at zero
        start_cropping_at_z = 0
    else:
        # take [0:cropped_resolution[0]]
        # Patients are given [x, y, z] when we say z it starts at zero (leg) to z (head).
        start_cropping_at_z = 2 * (z // 2 - (cropped_resolution[2] // 2))

    npad = ((off_set_x, off_set_x), (off_set_y, off_set_y), (2 * off_set_z, 0))

    # set zero value to the off set pixels mode='constant',
    data = np.pad(data, pad_width=npad, constant_values=0)

    # cropping to the given or set cropping resolution
    data = data[start_cropping_at_x:start_cropping_at_x + cropped_resolution[0],
           start_cropping_at_y:start_cropping_at_y + cropped_resolution[1],
           start_cropping_at_z:start_cropping_at_z + cropped_resolution[2]]

    return data


def save_nii_images(
        image: List[ndarray] = None, affine: ndarray = None, path_save: str = None, identifier: str = None,
        name: List[str] = None
        ):
    """ Save given images into the given directory. If no saving directory is given it will save into
    ./data/predicted/' directory.

    Args:
        image: data to save.
        affine: affine value.
        path_save: saving directory path.
        identifier: unique name of the case.
        name: name of the case to read.
    Returns:
        Saved image.
    """
    try:
        directory_exist(path_save)
    except:
        try:
            os.mkdir(path_save)
        except:
            if not os.path.exists("../data"):
                os.mkdir("../data")

            if not os.path.exists('../data/predicted/'):
                os.mkdir('../data/predicted/')

            path_save = '../data/predicted/'
    if identifier is not None:  # associate file name e.g. patient name
        dir = os.path.join(path_save, str(identifier))
    else:
        dir = path_save

    if not os.path.exists(dir):
        os.mkdir(dir)  # os.mkdir("./" + str(identifier))

    # print('saving it to %s\n' % dir)
    # .nii name, e.g. pet, gt, but if not given the base name of given directory
    if name is None:
        name = ['image_' + str(os.path.basename(dir)).split('.')[0]]

    if affine is None:
        affine = np.diag([4, 4, 4, 1])

    # image = np.flip(image, axis=-1)
    for select_image in range(len(image)):
        im_ = nib.Nifti1Image(image[select_image], affine)
        save_to = str(dir) + "/" + str(name[select_image])
        im_.to_filename(save_to)


def generate_mip_from_3D(given_3d_nii: ndarray = None, mip_axis: int = 0):
    """ Projects a 3D data into MIP along the selected MIP axis.

    Args:
        given_3d_nii: 3D array to project into MIP.
        mip_axis: Projecting axis.

    Returns:
        Returns MIP image.

    """
    if given_3d_nii is None:
        raise Exception("3D image not given")

    if not isinstance(given_3d_nii, list):
        given_3d_nii = np.array(given_3d_nii)

    mip = np.amax(given_3d_nii, axis=mip_axis)

    mip = np.asarray(mip)

    return mip


def transform_coordinate_space(modality_1, modality_2, mode='nearest'):
    """ Apply affine transformation on given data using affine from the second data.

    Adapted from: https://gist.github.com/zivy/79d7ee0490faee1156c1277a78e4a4c4
   Transfers coordinate space from modality_2 to modality_1
   Input images are in nifty/nibabel format (.nii or .nii.gz)


    Args:
        modality_1: reference modality.
        modality_2:  image modality to apply affine transformation.

    Returns:
        Returns affine transformed form of modality_2 using modality_1 as reference.

    """
    aff_t1 = modality_1.affine
    try:
        aff_t2 = modality_2.affine
    except:
        aff_t2 = aff_t1

    inv_af_2 = np.linalg.inv(aff_t2)
    out_shape = modality_1.get_fdata().shape

    # desired transformation
    T = inv_af_2.dot(aff_t1)

    # apply transformation
    transformed_img = scipy.ndimage.affine_transform(modality_2.get_fdata(), T, output_shape=out_shape, mode=mode)
    return transformed_img


# read PET and GT images
def read_pet_gt_resize_crop_save_as_3d_andor_mip(
        data_path: str = None, data_name: str = None, saving_dir: str = None, save_3D: bool = False, crop: bool = True,
        output_resolution: List[int] = None, desired_spacing: List[float] = None, generate_mip: bool = False
        ):
    """ Read pet and ground truth images from teh input data path. It also apply resize, and cropping operations.

    Args:
        data_path: directory to the raw 3D pet and ground truth .nii fies.
        data_name: unique the whole data identifier.
        saving_dir: directory path to save the processed images.
        save_3D: Save the processed 3D data or not.
        crop: apply cropping operations.
        output_resolution: desired output resolution.
        desired_spacing: required to be output spacing.
        generate_mip: project the processed 3D data into MIPs.

    Returns:
        Returns processed data.

    """
    if output_resolution is not None:
        # output resized and cropped image resolution
        rows, columns, depths = output_resolution
    else:  # default values
        # output resized and cropped image resolution=
        output_resolution = [128, 128, 256]

    if data_name is None:
        data_name = "unknown_data"

    '''
    get directory name  or patient id of the PET and gt images
    Assuming the file structure is :
   ---patient_id
         -- PET
            -- *.nii or .nii.gz
        -- gt
            -- *.nii or .nii.gz
    '''

    # check if the directory exist
    directory_exist(data_path)

    # by default the processed 3d and 2D MIP will be saved into the 'data' subdirectory, respectively with name tags
    # as '_default_3d_dir' and '_default_MIP_dir'

    def create_directory(directory_to_create: list):
        """

        :param directory_to_create:
        """
        for directory in directory_to_create:
            if not os.path.exists(directory):
                os.mkdir(directory)  # os.mkdir("./" + str(identifier))

    # if saving_dir is None:
    #     if not os.path.exists("../data"):
    #         os.mkdir("../data")
    #
    #     saving_dir_3d = "../data/" + str(data_name) + "_default_3d_dir"
    #
    #     # create directory in the data with name resized_cropped_MIP_default
    #     saving_dir_mip = "../data/" + str(data_name) + "_default_MIP_dir"
    #
    #     create_directory([saving_dir_3d, saving_dir_mip])
    #
    # # If the saving directory is given, a sub folder for the processed 3d and 2D MIP will be created
    # else:
    data_3d = str(data_name) + "_default_3d_dir_"
    data_mip = str(data_name) + "_default_MIP_dir"
    saving_dir_3d = os.path.join(saving_dir, data_3d)
    saving_dir_mip = os.path.join(saving_dir, data_mip)

    create_directory([saving_dir_3d, saving_dir_mip])

    # all patient ids
    case_ids = os.listdir(data_path)

    if not len(case_ids):  # reise exception if the directory is empty
        raise Exception("Directory %s is empty" % data_path)
    else:  # read the pet and gt images
        print("Continuing to read %d  cases" % len(case_ids))

    #  resize, to  given resolution
    if desired_spacing is None:
        desired_spacing = [4.0, 4.0, 4.0]

    # print where will be saved the 3d and mip
    print("\n")
    print(40 * "**==**")
    print("3D resized and cropped images will be saved to %s, if set save" % saving_dir_3d)
    print("Generated MIPs will be saved to %s, if set to save" % saving_dir_mip)
    print(40 * "**==**")
    print("\n")

    for image_name in tqdm(list(case_ids)):
        # if image_name in images:
        current_id = os.path.join(data_path, image_name)
        # read, resize, crop and save as 3D
        pet, gt = get_pet_gt(current_id)

        # resolution
        res_pet = pet.header.get_zooms()
        res_pet = tuple([float(round(res, 2)) for res in list(res_pet[:3])])
        affine = pet.affine
        print(f'Size of the PET input image: {np.asanyarray(pet.dataobj).shape}')

        # if there is a ground truth:
        if gt is not None:
            res_gt = gt.header.get_zooms()
            res_gt = tuple([float(round(res, 2)) for res in list(res_gt[:3])])

            # check if pet and gt are on the same spacing, otherwise gt could be generated from CT images
            if not res_pet[:3] == res_gt[:3]:  # assert (res_pet == res_gt)
                print("pet, and gt resolutions, respectively \n")
                print(res_pet, "\t", res_gt)
                warnings.warn("Pet and gt have different spacing, Alignment continue...")
                # apply affine transformation to move the gt to the PET space,
                # probably the gt was generated from the CT images
                gt = transform_coordinate_space(pet, gt, mode='constant')
                # raise Exception('Ground truth and pet images are not on the same spacing')
                gt[gt >= 1] = 1
                gt[gt < 1] = 0
            else:
                gt = np.asanyarray(gt.dataobj)

            """
               For remarc data the ground truth is inverted along the z-axis
               """
            if data_name == "remarc":
                gt = np.flip(gt, axis=-1)

            gt = resize_nii_to_desired_spacing(
                gt, data_spacing=res_pet, desired_spacing=desired_spacing, interpolation_order_value=0
                )

        pet = np.asanyarray(pet.dataobj)
        # if the given image has stacked as channel example one image in remarc : 175x175x274x2
        if pet.shape[-1] == 2:
            pet = pet[..., 0]

        # generate_mip_show(pet, gt, identifier=str(image_name))

        pet = resize_nii_to_desired_spacing(
            pet, data_spacing=res_pet, desired_spacing=desired_spacing, interpolation_order_value=3
            )

        '''
        if most data are with brain images at the very top, avoid cropping the brain images,instead crop from bottom
        or the leg part
        '''
        if gt is not None:
            if str(data_name).lower() == 'lnh':
                # flip left to right to mach lnh data to remarc
                gt = np.flip(gt, axis=-1)

        crop_zero_above_brain = True
        if crop_zero_above_brain:
            # remove all zero pixels just before the brian image
            xs, ys, zs = np.where(pet != 0)
            if len(zs):  # use zs for cropping the ground truth data also
                pet = pet[:, :, min(zs):]

        # if gt is None:
        #     generate_mip_show(pet, pet, identifier=str(image_name))
        # else:
        #     generate_mip_show(pet, gt, identifier=str(image_name))

        if crop:
            pet = crop_nii_to_desired_resolution(pet.copy(), cropped_resolution=output_resolution.copy())

            if gt is not None:
                # remove the zero values of pet image above the brain
                if len(zs):  # use zs for cropping the ground truth data also
                    gt = gt[:, :, min(zs):]

                gt = crop_nii_to_desired_resolution(gt.copy(), cropped_resolution=output_resolution.copy())

        # if gt is None:
        #     generate_mip_show(pet, pet, identifier=str(image_name))
        # else:
        #     generate_mip_show(pet, gt, identifier=str(image_name))

        if save_3D:
            # output image affine
            affine = np.diag([desired_spacing[0], desired_spacing[1], desired_spacing[2], 1])
            if gt is not None:
                save_nii_images(
                    [pet, gt], affine=affine, path_save=saving_dir_3d, identifier=str(image_name),
                    name=['pet', 'ground_truth']
                    )
            else:
                save_nii_images([pet], affine=affine, path_save=saving_dir_3d, identifier=str(image_name), name=['pet'])

        # generate Sagittal and coronal MIPs
        if generate_mip:
            for sagittal_coronal in range(2):
                pet_mip = generate_mip_from_3D(pet, mip_axis=int(sagittal_coronal))  # pet mip

                # assuming sagittal is on axis 0, and coronal is on axis 1
                if sagittal_coronal == 0:  # sagittal
                    naming_ = "sagittal"
                elif sagittal_coronal == 1:
                    naming_ = "coronal"

                if gt is not None:
                    gt_mip = generate_mip_from_3D(gt, mip_axis=int(sagittal_coronal))  # gt mip
                    # save the generated MIP
                    save_nii_images(
                        [pet_mip, gt_mip], affine, path_save=saving_dir_mip, identifier=str(image_name),
                        name=['pet_' + str(naming_), 'ground_truth_' + str(naming_)]
                        )
                else:
                    # save the generated MIP
                    save_nii_images(
                        [pet_mip], affine, path_save=saving_dir_mip, identifier=str(image_name),
                        name=['pet_' + str(naming_)]
                        )
    return saving_dir_mip


# Read .nii files using itk
if __name__ == '__main__':
    # how to run examples.
    # input_path = r"F:\Data\Remarc\REMARC/"
    # data_ = "remarc"
    #
    input_path = r"F:\Data\Vienna\No_ground_truth/"
    data_ = "LNH"
    saving_dir_mip = read_pet_gt_resize_crop_save_as_3d_andor_mip(
        data_path=input_path, data_name=data_, saving_dir=None, save_3D=True, crop=True,
        output_resolution=[128, 128, 256], desired_spacing=None, generate_mip=True
        )

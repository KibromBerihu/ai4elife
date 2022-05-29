# FAQ
### [FAQ](#faq)
- [Do I need CT images? ](#do-i-need-ct-images)
- [Where can I find the segmentation results, and computed biomarker features?](#where-can-i-find-the-segmentation-results-and-computed-biomarker-features)
- [Can I apply the proposed method to the 2D PET image-based segmentation](#can-i-apply-the-proposed-method-to-the-2d-pet-image-based-segmentation)
- [Can I run lfbnet on CPUs or GPUs?](#can-i-run-lfbnet-on-cpus-or-gpus)
- [Can I compare the predictability of the 2D biomarkers with 3D biomarkers?](#can-i-compare-the-predictability-of-the-2d-biomarkers-with-the-3d-biomarkers)
###  Do I need CT images? 
To run the program, you do not need CT scans. You only need PET scans in a nifty format where the PET images are coded
in SUV units.
 

### Where can I find the segmentation results and computed biomarker features?

The input PET nifty images are preprocessed into the shape of 128x128x256 and voxel size of 4x4x4. These preprocessed data
are saved under folder name  **"data_default_3d_dir_"**. If reference segmentations from an expert are provided along the
PET images with folder name **"GT"**, the preprocessing step applies the same as the PET images.
If the voxel sizes between the **GT** and PET images are different, it will transform the **GT** to the 
PET images. The **GT** might have been saved from CT spacing in these cases. Kindly check these
examples. 

From preprocessed 3D images, sagittal and coronal maximum intensity projections (MIP) are generated and saved under the folder name **data_default_mip_dir**.

The predicted segmentation results, the pet MIP images, and the reference ground truth data (if available) are saved 
the folder name **predicted**. 

Along the **"data_default_3d_dir_"**, **data_default_mip_dir**, and **predicted** subfolders, the calculated surrogate features
(**sTMVT and sDmax**) are saved in the EXCEL file. 


### Can I apply the proposed method to the 2D PET image-based segmentation?
The proposed method is optimized and applied for 2D MIP image-based segmentations. However, you can apply to 
the 2D images without projecting them into MIPs. The results will be saved as a 3D segmentation. Thus, you can directly apply the segmentation to the 2D-based PET images. Please use the virtual environment 
installation guide for these use cases mentioned in the [readme](https://github.com/KibromBerihu/ai4elife/blob/main/readme.md)  file. 
Kindly run the following command:

``python test_train_valid.py --input_dir path/to/input/data --output_dir path/to/output/data  --task [train, test, valid]``


### Can I run lfbnet on CPUs or GPUs? 
The whole package is tested on window 10 and ubuntu. If installing the dependencies is successful,
the package can run on either CPU or GPU. 
Please refer to [this]((https://github.com/KibromBerihu/ai4elife/blob/main/documentation/configuration)) for a more detailed selection of GPUs.
However, there is less parallelization and optimization of resource.


### Can I compare the predictability of the 2D biomarkers with the 3D biomarkers?
The main task of this package is to calculate the surrogate metabolic tumor burden (sTMTV) and dissemination (sDmax) features. However, if you have the reference segmentation in 3D, you can compare their correlation and survival predictability. To compute the metabolic tumor burden (TMTV) and tumor dissemitnaotn (Dmax) from
3D-images, please run the script `read_3D_nifti_mask_image_compute_TMTV_Dmax_save_as_csv_file.py` as follows:      
    

`python read_3D_nifti_mask_image_compute_TMTV_Dmax_save_as_csv_file.py  --input_dir  path/to/input/data  --output_dir path/to/output/data`


Note that the  structure of the input data should be:

    |-- main_folder                                             <-- The main folder or all patient folders (Give it any NAME)

    |      |-- patient_one                                      <-- Individual patient folder name with unique id (patient_one, can be any name)

    |           |-- patient_one.nii/nii.gz                       <-- Reference segmentation (ground truth image. The same name as the folder name.

    |      |-- patient_two                                      <-- Individual patient folder name with unique id (can be any name)                                       
    |           |-- patient_two.nii/nii.gz                       <-- Reference segmentation (ground truth image. The same name as the folder name.                                                                                                                                                                                                                             
                                                                                                                                                                                   
               .
               .
               .
    |      |-- patient_Nth                                     <-- Individual patient folder name with unique id (can be any name)                                       
    |           |-- patient_Nth.nii/nii.gz                       <-- Reference segmentation (ground truth image. The same name as the folder name.                                      
                                                                                                                                                                                      
                                                                                                                                                                                      
   
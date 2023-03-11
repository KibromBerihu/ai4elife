 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
 [![Docker build passing](https://img.shields.io/badge/docker%20build-passing-brightgreen)](https://github.com/KibromBerihu/ai4elife/blob/main/Dockerfile) 

# News!  
#### 3D version (stay tuned!)


## *AI4eLIFE: Artificial Intelligence for Efficient Learning-based Image Feature Extraction.*

#### <a name="introduction"> </a> [📑](https://jnm.snmjournals.org/content/early/2022/06/16/jnumed.121.263501) *Fully automated tumor lesions segmentation in whole-body PET/CT images using data-centric artificial intelligence and fully automatically calculating the clinical endpoints.*

***Introduction:***
Baseline 18F-FDG PET/CT image-driven features have shown predictive values in Diffuse Large B-cell lymphoma (DLBCL)
patients. Notably, total metabolic active tumor volume (TMTV) and tumor dissemination (Dmax) have shown predictive values to
characterize tumor burden and dissemination. However, TMTV and Dmax calculation require tumor volume 
delineation over the whole-body 3D 18F-FDG PET/CT images, which is prone to observer-variability and complicates using these quantitative features in clinical routine. In this regard, we hypothesized that tumor burden and spread could 
be automatically evaluated from only two PET Maximum Intensity Projections (MIPs) images corresponding to coronal and 
sagittal views, thereby easy the calculation and validation of these features. 

Here, we developed data-driven AI to calculate surrogate biomarkers for DLBCL patients automatically. Briefly, first, the (3D)
18F-FDG PET images were projected in the coronal and sagittal directions. The projected PET MIP images are then fed to 
an AI algorithm to segment lymphoma regions automatically. From the segmented images, the surrogate TMTV (sTMTV) and 
surrogate Dmax (sDmax) are calculated and evaluated in terms of predictions for overall survival (OS) and 
progression-free survival (PFS).

![flow-digaram](https://github.com/KibromBerihu/ai4elife/blob/main/images/graphical-abstract.JPG)

*Figure 1: Flow diagram of the proposed data-centric AI to measure prognostic biomarkers automatically.*

***Results:***
Tested on an independent testing cohort (174 patients), the AI yielded a 0.86 median Dice score (IQR: 0.77-0.92), 87.9%
(IQR: 74.9.0%-94.4%) sensitivity, and 99.7% (IQR: 99.4%-99.8%) specificity. The PET MIP AI-driven surrogate biomarkers (sTMTV) and sDmax were highly correlated to the 3D 18F-FDG PET-driven biomarkers
(TMTV and Dmax) in both the training-validation cohort and the independent testing cohort. These PET MIP AI-driven 
features can be used to predict the OS and PFS in DLBCL patients, equivalent to the expert-driven 3D features. 

***Deep learning Model:*** 
We adapted the deep learning-based robust medical image segmentation method [LFBNet](https://doi.org/10.1109/TMI.2021.3060497).
Please refer to the [paper](https://doi.org/10.1109/TMI.2021.3060497) 
for details, and cite the paper if you use lfbnet for your research. 

[comment]: <![img_7.png](img_7.png)>

***Integrated framework:***
The whole pipeline, including the generation of PET MIPs, automatic segmentation, and sTMTV and sDmax calculation, is developed 
for a use case on personal/desktop computers or clusters. It could highly facilitate the analysis of PET MIP-based features 
leading to the potential translation of these features into clinical practice. 

Please refer to the paper for details and cite the paper if you use LFBNet for your research. 

### Table of contents  
- [Summary](#introduction)
- [Table of Contents](#table-of-contents)
- [ Required folder structure](#-required-folder-structure)
- [Installation](#installation)
- [Usage](#-usage)
  - [Easy use: testing mode](#easy-use-testing-mode)
  - [Transfer learning mode: development](#transfer-learning-mode-developmenttranserlearning)
- [Results](#-results)
- [FAQ](#-faq)
- [Citations](#-citations)
- [Adapting LFBNet for other configurations or segmentation tasks](#-how-to-configure-an-extended-lfbnet-for-other-2d-based-medical-image-segmentation)
- [Useful resources](#useful-resources) 
- [Acknowledgements](#-acknowledgments)

## 📁 Required folder structure
Please provide all data in a single directory. The method automatically analyses all given data batch-wise. 

To run the program, you only need PET scans (CT is not required) of patients in nifty format, where the PET images are coded in SUV units. If your images have already been segmented, you can also provide the mask (ground truth (gt)) as a binary image in nifty format. Suppose you provided ground truth (gt) data; it will print the dice, sensitivity, and specificity metrics between the reference segmentation by the expert (i.e., gt) and the predicted segmentation by the model. If the ground truth is NOT AVAILABLE, the model will only predict the segmentation. 

A typical data directory might look like:


    |-- main_folder                                             <-- The main folder or all patient folders (Give it any NAME)

    |      |-- parent folder (patient_folder_1)             <-- Individual patient folder name with unique id
    |           |-- pet                                     <-- The pet folder for the .nii suv file
                     | -- name.nii or name.nii.gz            <-- The pet image in nifti format (Name can be anything)
    |           |-- gt                                      <-- The corresponding ground truth folder for the .nii file  
                     | -- name.nii or name.nii.gz            <-- The ground truth (gt) image in nifti format (Name can be anything)
    |      |-- parent folder (patient_folder_2)             <-- Individual patient folder name with unique id
    |          |-- gt                                     <-- The pet folder for the .nii suv file
                    | -- name.nii or name.nii.gz            <-- The pet image in nifti format (Name can be anything)
    |         |-- pet                                      <-- The corresponding ground truth folder for the .nii file  
                    | -- name.nii or name.nii.gz            <-- The ground truth (gt) image in nifti format (Name can be anything)
    |           .
    |           .
    |           .
    |      |-- parent folder (patient_folder_N)             <-- Individual patient folder name with unique id
    |           |-- pet                                     <-- The pet folder for the .nii suv file
                    | -- name.nii or name.nii.gz            <-- The pet image in nifti format (Name can be anything)
    |           |-- gt                                      <-- The corresponding ground truth folder for the .nii file  
                    | -- name.nii or name.nii.gz            <-- The ground truth (gt) image in nifti format (Name can be anything)


 **Note:** the folder name for PET images should be `pet` and for the ground truth `gt`. All other folder and sub-folder names could be anything. 

## ⚙️  Installation <a name="installation"> </a>
  
Please read the documentation before opening an issue! 

<font size='4'> Download/clone code to your local computer </font> 


    - git clone https://github.com/KibromBerihu/ai4elife.git
   
    - Alternatively:
      1. go to https://github.com/KibromBerihu/ai4elife.git >> [Code] >> Download ZIP file.
 
      


   1) <font size ="4">To install in virtual environment </font>  
       1) We recommend you to create virtual environment. please refer to [THIS](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) regarding how to create a virtual environment using 
         conda.  
       2) Open terminal or Anaconda Prompt <br/><br>
       3) Change the working directory to the downloaded and unzipped ai4elife folder <br/><br>
       4) Create the virtual environment provided in the requirements.yaml: 
      
           `conda env create -f environment.yml`
      <br/><br>
       5) If you choose to use a virtual environment, the virtual environment must be activated before executing any script:
      
           `conda activate myenv`
      <br/><br>
       6) Verify the virtual environment was installed correctly:
   
            `conda info --envs`
      
           <font size='2'>  If you can see the virtual environment with a name 'myenv', well done, the virtual environment and dependencies are installed successfully. </font>

         

   2) <font size ="4"> Using docker image: building image from docker file [REPRODUCIBLE] </font> <br/><br>
      
      1) Assuming you already have [docker desktop](https://www.docker.com/) installed. For more information, kindly refer to [THIS](https://docs.docker.com/). 
      
      2) Make sure to change the directory to the downloaded and unzipped ai4elife directory. 
      <br/><br>
      3) Run the following commands to create a docker image with the name <DockerImageName>:<Tag>'
      <br/><br>

         `docker build -t <DockerImageName>:<Tag> .`

 
## 💻 Usage
This package has two usages. 
The first one is to segment tumor regions and then calculate the surrogate biomarkers such as sTMTV and sDmax on the given test dataset using the pre-trained weights, named as "easy use case". 
The second use case is transfer learning or retraining from scratch on your own dataset.

 
### [Easy use: testing mode](#virtual) <a name="easy-use-testing-mode"> </a> 

Please make sure that you organized your data as in the [Required folder structure](#-required-folder-structure).
### For reproducibility and better accuracy, please use OPTION 2. 
1. **Option 1:** Using the virtual environment: <br/><br>
    1. Change to the source directory: `cd  path/to/ai4elife/` <br/><br>
    2. Activate the virtual environment:    `conda activate myenv` <br/><br>
 
    3. Run:  `python test_env.py  --input_dir path/to/input/  --output_dir path/to/output/` 
   <br/><br>
2. **Option 2:** Using the docker: <br/><br>
   
    Option 1:`run_docker_image.bat path/to/input path/to/output  <docker_image_name> <Tag>  <container_id>`
    <br/><br>
    Option 2: `docker run -it --rm --name <container_id> -v path/to/input/:/input -v path/to/output/:/output <docker_image_name>:<Tag>`
    <br/><br>
 
### [Transfer learning mode: development](#transerlearning) <a name="transfer-learning-mode-developmenttranserlearning"> </a>  
 
To apply transfer learning by using the trained weights or training the deep learning method from scratch,
we recommend following the virtual environment-based [installation](#installation) option.

Run the following commands for activating the virtual environment, and then training, validating, and testing of the proposed model on your own dataset.

1. Activate the virtual environment:
   `conda activate myenv`
<br/><br>
2. To [train](#train) the model from a new dataset, change to the `ai4elife/src` directory: <br/><br>
   
   `python train.py --input_dir path/to/training_validation_data/  --data_id <unique_data_name> --task <train>`
<br/><br>
3. To [evaluate](#evaluate) on the validation data: <br/><br>
    `python train.py --input_dir path/to/validation_data/  --data_id <unique_data_name> --task <valid>`
<br/><br>

**Note:** You can also **configure** the deep learning model for **parameter and architectural search**. Please refer to the documentation
[configuration](architectural_and_parameter_search.md). Briefly, you can apply different features, kernel size in the convolution, depth of the neural networks, and other hyperparameters values. The segmentation 
model is designed in easy configurable mode. 
   
## 📈 Results

- Two intermediate folders will be generated.

  - The resized and cropped 3D PET, and corresponding ground truth  Nifti images are saved under the folder name:
                  
      ```../output/data_default_3d_dir```, and 
  
  - The generated corresponding sagittal and coronal images are saved in the folder name       
``../output/data_default_mip_dir``.
  
  - For simplicity, the coronal PET MIP images are `pet_coronal.nii`, sagittal as `pet_sagittal.nii`, and corresponding ground truth as `ground_truth_coronal.nii`, and `ground_truth_sagittal.nii`, respectively.
  
  - NOTE: if there is no ground truth, it will only generate the coronal and sagittal PET MIPs. 
  Kindly check if these generated files are in order.
  
  
- Predicted results including predicted segmentation masks and calculated surrogate biomarekrs (sTMTV and sDmax) will be saved into the folder `output/.*`. 

 - Predicted masks are saved under the folder name `output/predicted/.*`. The predicted masks are indicated with the keyword `predicted` in the file name, and the input PET images are indicated with the keyword `pet`. For example, `patient_id_predicted.nii` for predicted mask of `patient_id` and `patient_id_pet.nii` for the PET images. If the ground truths are given, they are saved with the name `patient_id_ground_truth.nii`. Each .nii file has both sagittal and coronal views Each .nii file has both sagittal and coronal views concatenated.  
 
- Surrogate biomarkers (sTMTV and sDmax) will be automatically calculated and saved as an EXCEL file inside the folder output/*.csv. Two EXCEL files will be saved. The first one constitutes computed surrogate biomarkers calculated from the segmentation masks predicted from AI with an indicator `predicted` in the file name. The second EXCEL file would constitute the surrogate biomarkers computed from the reference segmentation masks (i.e., ground truth) from the expert (if available) with an indicator `ground_truth` in the file name. In addition to the `predicted` and `ground truth` indicator names, the CSV file's name also constitutes an automatically generated month, year, and the processing time.


## 🙋 FAQ
Please visit the [FAQ](./documentation/FAQ.md) samples before creating an issue. 

## 📖 Citations 
Please cite the following papers if you use this package for your research:
```
Girum KB, Rebaud L, Cottereau A-S, et al. 18 F-FDG PET maximum intensity projections and artificial intelligence: a win-win combination to easily measure prognostic biomarkers in DLBCL patients. J Nucl Med. June 2022:jnumed.121.263501. DOI: https://doi.org/10.2967/jnumed.121.263501 
```
```
 Girum KB, Créhange G, Lalande A. Learning with context feedback loop for robust medical image segmentation. IEEE Transactions on Medical Imaging. 2021 Feb 19;40(6):1542-54. DOI: https://doi.org/10.1109/TMI.2021.3060497
```

## 💭 How to configure an extended LFBNet for other 2D-based medical image segmentation? 
LFBNet is provided as a configurable network for 2D image-based multi-class and single-class segmentations.
Please refer to [THIS](./documentation/tutorial/jupyter_notebook_step_by_step_illustration.ipynb) guide. 


## 💁️<a name="useful-resources"> </a> Useful resources
- The detailed step-by-step for preprocessing, dataset split into training and validation cohorts, and visualization of results are demonstrated in the [jupyter_notebook_step_by_step_illustration.ipynb](./documentation/tutorial/jupyter_notebook_step_by_step_illustration.ipynb).

## 🙏 Acknowledgments
We thank you [the reader].  



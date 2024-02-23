# Overview

This README contains all information on how to run 24BrainMRI_Preprocessing(24BMP for short), a preprocessing pipeline for *Development of an early Prognosis prediction model for Central Nervous System Infections using AI-based analysis of Brain MRIs*.

![](/images/pipeline_diagram.png)

Above is a diagram for the 24BMP Pipeline. The 24BMP pipeline consists of three main parts for **Resampling**, **Segmentation** and **Brainpart Extraction**. The **Resampling** Part relies entirely on the resampling method used in **lab2im**, and the **Segmentation** Part on FastSurferCNN from **FastSurfer**. Both libraries are cited in the bottom of this README.

# Getting Started

## Installation
To install 24BMP, clone this repository to your desired location using below code :

```bash
git clone https://github.com/choiyh592/24BrainMRI_Preprocessing_Dev.git
cd 24BrainMRI_Preprocessing_Dev
```

### Requirements
24BM was developed using Python 3.10.4. Thus a Python version of 3.10.4 is recommended.

You can install all requirements by executing the following code.
* For Windows or Linux : 
```bash
 python3 -m pip install -r requirements.txt
```

* For Mac : 
```bash
python3 -m pip install -r requirements.mac.txt
```

## Usage
The Syntax for running the entire pipeline is :
```
python3 24BrainMRI_Preprocessing/main.py \
    --inputtxt /path/to/your/input/filepaths/text/file \
    --sidtxt /path/to/your/patient/ids/text/file \
    --outputdir /path/to/your/output/directory \
    --device your_device \
    --va False
```
To execute the entire pipeline, you can run the above code in the terminal after navigating to the root directory of your local repository.

The execution takes about 6 minutes per MRI, assuming that FastSurferCNN is run on GPU. Using a GPU is highly recommended.

### Flags
24BMP utilizes the ```argparse``` module from python to interpret the flags(arguments). You can view the descriptions by running : 
```
python 24BrainMRI_Preprocessing/main.py --help
```
Below are descriptions for the flags.
* `--inputtxt`(str): Text File containing T1 MRI paths for all subjects. There should be only one path per line. An example is provided below.
* `--outputdir`(str): Path for a single directory to contain all outputs. An example is provided below.
* `--sidtxt`(str): Text File containing the subject ID to use. There should be only one ID per line. \
Should be in the same order as the --inputtxt file. Should have the same number of lines as the `--inputtxt` file. Should be unique. \
An example is provided below.
* `--device`(str): Select device to run FastSurferCNN inference on: cpu, or cuda(= Nvidia GPU) or specify a certain GPU (e.g. cuda:1). ```auto``` by default.
* `--va`(bool) : If true, volumetric analyses on all segmented files are performed. False by default.

### Input Requirements & Recommendations
24BMP Takes 4 Arguments: `--inputtxt`, `--sidtxt`, `--outputdir`, `--device`.

`--inputtxt` : String containing the **Absolute Path** to Text file containing the **Absolute Paths** of the input T1 MRI Data. Below is an example.
```
--inputtxt /home/yhchoi/textfiles/niftipaths.txt
```
Below is an example illustrating the content format of the file.
```
/home/yhchoi/NIFTI/T1/01.nii.gz
/home/yhchoi/NIFTI/T1/02.nii.gz
```
Note that order matters : The order of data in the final results will depend on the order of the filepaths in `--inputtxt`.

`--sidtxt` : String containing the **Absolute Path** to Text file containing the **Unique** IDs for the resampled T1 MRI Data. Below is an example.
```
--sidtxt /home/yhchoi/textfiles/patientIDs.txt
```
Below is an example illustrating the content format of the file.
```
1_11112222
2_22223333
```
In this format, each line consists of an index followed by an underscore and an ID.
This indexing method ensures uniqueness and aids in maintaining the order of file paths. \
Note that the order of the filepaths should be the same as in `--inputtxt`. \
We recommend that you index the IDs like above (indexed and separated with an underscore) to assure uniqueness.

`--outputdir` : String containing the **Absolute Path** where you wish to contain the outputs. Below is an example.
```
/home/yhchoi/project/preprocessing_data
```

`--device` : String containing the name of the device to perform FastSurferCNN Inference on. We highly recommend using a GPU('cuda').

`--va` : This option decides whether to conduct volumetric analysis. Can be `True` or `False`. If `True`, the outcomes will be saved as `__volumetic_analysis.csv` in the `/test_files` directory within the output folder. Patient volumes are recorded in cubic millimeters (mm<sup>3</sup>) If `False`(Default), volumetic analysis is skipped. Volumetric analysis takes about 30 seconds per image.

### Outputs
The `--outputdir` flag determines the (absolute) location of the outputs.

4 subdirectories will be created within the output directory : `resampled`, `segmentations`, `extractions`, `text_files`.
* `resampled` : Subdirectory containing the resampled MRI files.
* `segmentations` : Subdirectory containing the segment masks and conformed MRI files. 
Segmentation of each patient is stored in subdirectories with the patient ID as the name. \
Below is the FastSurfer documentation for the segmentation outputs.

    | directory   | filename                      | module    | description |
    |:------------|-------------------------------|-----------|-------------|
    | mri         | aparc.DKTatlas+aseg.deep.mgz  | asegdkt   | cortical and subcortical segmentation|
    | mri         | aseg.auto_noCCseg.mgz         | asegdkt   | simplified subcortical segmentation without corpus callosum labels|
    | mri         | mask.mgz                      | asegdkt   | brainmask|
    | mri         | orig.mgz                      | asegdkt   | conformed image|
    | mri/orig    | 001.mgz                       | asegdkt   | original image|
    | .           | qc_log.log                    | asegdkt   | quality check logfile : will be empty if the qc process is successful|
    | .           | seg_log.log                   | asegdkt   | segmentation logs|

* `extractions` : Subdirectory containing the extracted, cropped and padded MRI files and a tensor dataset containing the files. \
The order of files in the tensor dataset corresponds to the sequential order specified in the `--inputtxt` file.
Below is the documentation for the extraction outputs.

    | directory   | description                   |
    |:------------|-------------------------------|
    | {BRAINPART}_EXTRACT | Directory containing extracted MRI files of specific brainpart {BRAINPART}|
    | {BRAINPART}_CROPPED | Directory containing extracted MRI files of {BRAINPART} cropped to minimal size|
    | {BRAINPART}_PADDED  | Directory containing cropped MRI files of {BRAINPART} padded to match sizes|
    | {BRAINPART}_TENSORS | Directory containing tensor dataset containing the padded MRI files of {BRAINPART}|

    Brain parts categorized as left and right were placed into separate subdirectories named /left and /right.
    The left and right images were merged into a single tensor dataset.

* `text_files` : Subdirectory containing text files with straightforward names, which contains paths to created files.
The order of paths in the text file corresponds to the sequential order specified in the `--inputtxt` file.\
CSV files containing the outcomes of quality checks and volumetric analyses are also provided in this directory.\
The `__qc_failed.csv` file contains paths to files that failed the quality check. The files in the csv should have faulty segmentations. \
The `__volumetric_analysis.csv` file contains volumetric assessments for each patient, identified by their unique patient IDs provided in `--sid`.

Below is an example output demonstrating the utilization of the flags provided above.
```
outputdir
├── extractions
│   ├── 3RD_VENTRICLE_CROPPED
│   │   ├── 1_11112222_3rd_ventricle.nii.gz
│   │   └── 2_22223333_3rd_ventricle.nii.gz
│   ├── 3RD_VENTRICLE_EXTRACT
│   │   ├── 1_11112222_3rd_ventricle.nii.gz
│   │   └── 2_22223333_3rd_ventricle.nii.gz
│   ├── 3RD_VENTRICLE_PADDED
│   │   ├── 1_11112222_3rd_ventricle.nii.gz
│   │   └── 2_22223333_3rd_ventricle.nii.gz
│   ├── 3RD_VENTRICLE_TENSORS
│   │   └── 3rd_ventricle_dataset.pt
│   ...
│   ├── VENTRALDC_PADDED
│   │   ├── left
│   │   │   ├── 1_11112222_ventraldc_left.nii.gz
│   │   │   └── 2_22223333_ventraldc_left.nii.gz
│   │   └── right
│   │       ├── 1_11112222_ventraldc_right.nii.gz
│   │       └── 2_22223333_ventraldc_right.nii.gz
│   └── VENTRALDC_TENSORS
│       └── ventraldc_dataset.pt
├── resampled
│   ├── 1_11112222_resampled.nii.gz
│   └── 2_22223333_resampled.nii.gz
├── segmentations
│   ├── 1_11112222
│   │   ├── mri
│   │   │   ├── orig
│   │   │   │   └── 001.mgz
│   │   │   ├── aparc.DKTatlas+aseg.deep.mgz
│   │   │   ├── aseg.auto_noCCseg.mgz
│   │   │   ├── mask.mgz
│   │   │   └── orig.mgz
│   │   ├── qc_log.log
│   │   └── seg_log.log
│   └── 2_22223333
│       ├── mri
│       ...
└── text_files
    ├── __qc_failed.csv
    ├── __volumetric_analysis.csv
    ├── _conformed_paths.txt
    ├── _resampled_paths.txt
    ├── _segmented_paths.txt
    ├── nifti_3rd_ventricle_cropped_paths.txt
    ├── nifti_3rd_ventricle_padded_paths.txt
    ├── nifti_3rd_ventricle_padded_paths.txt
    ...
    ├── nifti_ventraldc_left_paths.txt
    └── nifti_ventraldc_right_paths.txt
```

# Citations

### Lab2Im

```
A Learning Strategy for Contrast-agnostic MRI Segmentation
Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
*contributed equally
MIDL 2020
```
[Link](https://github.com/BBillot/lab2im/tree/master)

### FastSurfer

```
FastSurfer - A fast and accurate deep learning based neuroimaging pipeline
Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M
NeuroImage 219 (2020), 117012

FastSurferVINN: Building Resolution-Independence into Deep Learning Segmentation Methods - A Solution for HighRes Brain MRI.
Henschel L*, Kuegler D*, Reuter M.
*co-first
NeuroImage 251 (2022), 118933
```
[Link](https://github.com/deep-mi/FastSurfer)

## References

If you use this for research publications, please consider citing:
```
Development of an early Prognosis prediction model for Central Nervous System Infections using AI-based analysis of Brain MRIs
TBD
```
along with the above citations.

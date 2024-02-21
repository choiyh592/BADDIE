# Overview

This README contains all information on how to run 24BrainMRI_Preprocessing(24BMP for short), a preprocessing pipeline for *Development of an early Prognosis prediction model for Central Nervous System Infections using AI-based analysis of Brain MRIs*.

![](/images/pipeline_diagram.png)

Above is the 24BMP Pipeline. The 24BMP pipeline consists of three main parts for **Resampling**, **Segmentation** and **Brainpart Extraction**. The **Resampling** Part relies entirely on the resampling method used in **lab2im**, and the **Segmentation Part** on FastSurferCNN from **FastSurfer**. Both libraries are cited in the bottom of the README.

# Getting Started

## Installation
To install 24BMP, clone this repository to your desired location using below code :

```bash
git clone https://github.com/choiyh592/24BrainMRI_Preprocessing_Dev.git
cd 24BrainMRI_Preprocessing_Dev
python3 -m pip install -r requirements.txt
```
You can install all requirements by executing the 
## Usage
The Syntax for running the entire pipeline is :
```
python 24BrainMRI_Preprocessing/main.py 
    --inputtxt /path/to/your/input/filepaths/text/file \
    --resampledtxt /path/to/your/resampled/filepaths/text/file \
    --outputdir /path/to/your/output/directory \
    --sidtxt /path/to/your/patient/ids/text/file \
    --device your_device
```
To execute the entire the pipeline, you can run the above code in the terminal after navigating to the root directory of your local repository.

The execution takes about 6 minutes per MRI, assuming that FastSurferCNN is run on GPU. Using a GPU is highly recommended.

### Flags
24BMP utilizes the ```argparse``` module from python to interpret the flags(arguments). You can view the descriptions by running : 
```
python 24BrainMRI_Preprocessing/main.py --help
```
Below are descriptions for the flags.
* `--inputtxt`: Text File containing T1 MRI paths for all subjects. There should be only one path per line. An example is provided below.
* `--resampledtxt`: Text File containing paths for resampled MRIs of all subjects. There should be only one path per line. Should be in the same order as the --inputtxt file. Should have the same number of lines as the --inputtxt file. An example is provided below.
* `--outputdir`: Path for a single directory to contain all outputs. An example is provided below.
* `--sidtxt`: Text File containing the subject ID to use. There should be only one ID per line. Should be in the same order as the --inputtxt file. Should have the same number of lines as the --inputtxt file. An example is provided below.
* `--device`: Select device to run FastSurferCNN inference on: cpu, or cuda(= Nvidia GPU) or specify a certain GPU (e.g. cuda:1). ```auto``` by default.

### Input Requirements & Recommendations
24BMP Takes 5 Arguments: `--inputtxt`, `--resampledtxt`, `--outputdir`, `--sidtxt`, `--device`.

# What It Does

## Resampling

## Segmentation

## Brainpart Extraction

## Citations

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

If you use this for research publications, please cite:
```
Development of an early Prognosis prediction model for Central Nervous System Infections using AI-based analysis of Brain MRIs
TBD
```
along with the above citations.

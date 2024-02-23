import nibabel as nib
import numpy as np
import pandas as pd
from Seg2Seg.src.labels import labels_lookup
from Seg2Seg.src.imageutils import calculate_volume

def calculate_volumes(nifti_path, mask_path, sids_path, txt_file_path):
    """
    This function calculates the volume of each brain part in the segmentation mask
    Args:
    nifti_path: str: path to the NIfTI image
    mask_path: str: path to the segmentation mask
    Returns:
    result_df: pd.DataFrame: DataFrame containing the brain part and its corresponding volume
    """
    volume_dict = {'ID':[]}

    for key, _ in labels_lookup.items():
        lookup = key.replace('-', '_')
        volume_dict[lookup] = []

    nifti_files = read_file(nifti_path)
    mask_files = read_file(mask_path)
    sids = read_file(sids_path)

    for nifti_path, mask_path, sid in zip(nifti_files, mask_files, sids):
        volume_dict['ID'].append(sid)
        for key, value in labels_lookup.items():
            lookup = key.replace('-', '_')
            volume = calculate_volume(nifti_path, mask_path, value)
            volume_dict[lookup].append(volume)
        
    result_df = pd.DataFrame(volume_dict)
    result_df.to_csv(txt_file_path / '__volumetic_analysis.csv', index=False)

def read_file(file_path):
    # Open the file in read mode
    with open(file_path, "r") as file:
        lines = file.readlines()  # Read all lines into a list

    # Remove newline characters from each line and store in a new list
    lines = [line.strip() for line in lines]

    # Print the list of lines
    return lines
import torch
import pandas as pd
import nibabel as nib
from Seg2Seg.src.labels import labels, label_lookups
from Seg2Seg.src.imageutils import quality_check

def segmentations_qc(srcdir):
    """
    This function runs quality check on the segmented files
    Args:
    srcdir: str: source directory
    Returns:
    qc_filepath: str: path to the quality check file
    """

    srcdir = str(srcdir)
    failed_qc_parts = []
    failed_qc_filenames = []
    for part_to_segment in labels:
        
        label_for_left_part = label_lookups(part_to_segment, 'left')
        label_for_right_part = label_lookups(part_to_segment, 'right')
        part_to_segment = part_to_segment.replace('-', '_')

        if label_for_left_part != label_for_right_part :
            cropped_left_txt_path = srcdir + '/' + 'nifti_'+ part_to_segment.lower() + '_left_cropped_paths.txt'
            cropped_right_txt_path = srcdir + '/' + 'nifti_'+ part_to_segment.lower() + '_right_cropped_paths.txt'
            failed_qc_list_left = quality_check(cropped_left_txt_path)
            failed_qc_part_left = [f'{part_to_segment}_left' for i in range(len(failed_qc_list_left))]
            failed_qc_list_right = quality_check(cropped_right_txt_path)
            failed_qc_part_right = [f'{part_to_segment}_right' for i in range(len(failed_qc_list_right))]
            failed_qc_parts += failed_qc_part_left + failed_qc_part_right
            failed_qc_filenames += failed_qc_list_left + failed_qc_list_right

        elif label_for_left_part == label_for_right_part :
            cropped_txt_path = srcdir + '/' + 'nifti_'+ part_to_segment.lower() + '_cropped_paths.txt'
            failed_qc_list = quality_check(cropped_txt_path)
            failed_qc_part = [f'{part_to_segment}' for i in range(len(failed_qc_list))]
            failed_qc_parts += failed_qc_part
            failed_qc_filenames += failed_qc_list
    
    df = pd.DataFrame({'filename': failed_qc_filenames, 'part': failed_qc_parts})

    qc_filepath = srcdir + '/' + '__qc_failed.csv'
    df.to_csv(qc_filepath, index=False)

    return qc_filepath
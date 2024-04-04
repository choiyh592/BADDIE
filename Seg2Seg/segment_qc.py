import torch
import pandas as pd
from Seg2Seg.src.labels import labels, label_lookups
from Seg2Seg.src.imageutils import quality_check, process_nifti_files, process_nifti_files_nLR

def retrieve_patient_idx(line):
    """
    This function retrieves the patient index from the file path
    Args:
    line: str: file path
    Returns:
    patient_idx: str: patient index
    """
    # patient_idx = line.strip("\n").split('/')[-1].split('_')[0] + '_' + line.strip("\n").split('/')[-1].split('_')[1]
    patient_idx = line.strip("\n").split('/')[-1].split('_')[0]
    return patient_idx

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
            failed_qc_part_left = [f'{part_to_segment}_left' for _ in range(len(failed_qc_list_left))]
            failed_qc_list_right = quality_check(cropped_right_txt_path)
            failed_qc_part_right = [f'{part_to_segment}_right' for _ in range(len(failed_qc_list_right))]
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

def remove_lines_from_text(text_file, lines_to_delete):
    """
    This function removes the filepaths of the failed QC files from the text files
    Args:
    text_file: str: path to the text file
    lines_to_delete: list: list of lines to delete
    Returns:
    None
    """
    counter = 0
    with open(text_file, "r") as f:
        lines = f.readlines()
    with open(text_file, "w") as f:
        for line in lines:
            patient_idx = retrieve_patient_idx(line)
            if patient_idx not in lines_to_delete:
                f.write(line)
            else:
                print(f'Removing image of patient {patient_idx} from {text_file}...')
                counter += 1
    print(f'{counter} images in total removed from {text_file}...')

def remove_qcfailed(outputdir):
    """
    This function runs quality check on the segmented files
    Args:
    outputdir: str: output directory
    Returns:
    None
    """

    srcdir = outputdir + '/' + 'text_files'
    excdir = outputdir + '/' + 'extractions'
    qc_file = srcdir + '/' + '__qc_failed.csv'
    va_file = srcdir + '/' + '__volumetric_analysis.csv'

    print('Processing Started!')
    df = pd.read_csv(qc_file)
    failed_qc_filenames = df['filename'].tolist()
    failed_qc_indices = [retrieve_patient_idx(filename) for filename in failed_qc_filenames]
    failed_qc_indices = list(set(failed_qc_indices))

    for part_to_segment in labels:
        label_for_left_part = label_lookups(part_to_segment, 'left')
        label_for_right_part = label_lookups(part_to_segment, 'right')
        print(f'Processing {part_to_segment}...')
        part_to_segment = part_to_segment.replace('-', '_')

        if label_for_left_part != label_for_right_part :

            # Dirs
            padded_left_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_left_padded_paths.txt'
            padded_right_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_right_padded_paths.txt'

            # Remove the failed files from the text files
            print('Removing failed files from the text files...')
            remove_lines_from_text(padded_left_path, failed_qc_indices)
            remove_lines_from_text(padded_right_path, failed_qc_indices)
            remove_lines_from_text(va_file, failed_qc_indices)

            tensordir = excdir + '/' + part_to_segment.upper() + '_TENSORS/'

            tensor_stack = process_nifti_files(padded_left_path, padded_right_path)
            save_path = tensordir + part_to_segment.lower() + '_dataset.pt'
            torch.save(tensor_stack, save_path)
            print(f'All images saved as tensors at {save_path}')
            print(f'Processing of {part_to_segment} Complete!')

        elif label_for_left_part == label_for_right_part :

            padded_path = srcdir + '/' + 'nifti_'+ part_to_segment.lower() + '_padded_paths.txt'

            # Remove the failed files from the text files
            print('Removing failed files from the text files...')
            remove_lines_from_text(padded_path, failed_qc_indices)
            remove_lines_from_text(va_file, failed_qc_indices)
            tensor_stack = process_nifti_files_nLR(padded_path)

            tensordir = excdir + '/' + part_to_segment.upper() + '_TENSORS/'
            save_path = tensordir + '/' + part_to_segment.lower() + '_dataset.pt'
            torch.save(tensor_stack, save_path)
            print(f'All images saved as tensors at {save_path}')
            print(f'Processing of {part_to_segment} Complete!')

    print('Processing Complete!')
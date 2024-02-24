import torch
import numpy as np
import nibabel as nib
from Seg2Seg.src.labels import labels, label_lookups
from Seg2Seg.src.imageutils import crop, pad, process_nifti_files, process_nifti_files_nLR
from Seg2Seg.src.createdirs import createdirs_LR, createdirs_NLR

# labels
def process_images(resampled_file_path, mask_file_path, srcdir, outputdir, num_of_files = 0):
    """
    This function extracts the images from the resampled_file_path and mask_file_path and saves them as tensors in the outputdir
    Args:
    resampled_file_path: str: path to the resampled nifti file
    mask_file_path: str: path to the mask nifti file
    srcdir: str: path to the directory containing the nifti files
    outputdir: str: path to the directory to save the tensors
    num_of_files: int: number of files to process : needed for efficient logging (non-manadatory)
    """

    resampled_file_path = str(resampled_file_path)
    mask_file_path = str(mask_file_path)
    srcdir = str(srcdir)
    outputdir = str(outputdir)

    for part_to_segment in labels:
        label_for_left_part = label_lookups(part_to_segment, 'left')
        label_for_right_part = label_lookups(part_to_segment, 'right')

        if label_for_left_part != label_for_right_part :

            # Dirs to create
            part_to_segment = part_to_segment.replace('-', '_')
            segment_left_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_left_paths.txt'
            segment_right_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_right_paths.txt'
            cropped_left_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_left_cropped_paths.txt'
            cropped_right_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_right_cropped_paths.txt'
            padded_left_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_left_padded_paths.txt'
            padded_right_path = srcdir + '/nifti_'+ part_to_segment.lower() + '_right_padded_paths.txt'

            ###############################################
            # OS filepaths & txt files configurations
            ###############################################

            tensordir = createdirs_LR(mask_file_path, outputdir, part_to_segment, segment_left_path, segment_right_path, cropped_left_path,
                            cropped_right_path, padded_left_path, padded_right_path)

            ###############################################
            # Extract Images
            ###############################################

            def process_images(mri_file, mask_file, output_left_file, output_right_file):
                with open(mri_file, 'r') as mri_paths, open(mask_file, 'r') as mask_paths, open(output_left_file, 'r') as output_left, open(output_right_file, 'r') as output_right:
                    print(f'Starting Extraction of Part : {part_to_segment}...')
                    count = 0
                    for mri_path, mask_path, output_left_path, output_right_path in zip(mri_paths, mask_paths, output_left, output_right):
                        try:
                            mri_path = mri_path.strip()
                            mask_path = mask_path.strip()
                            output_left_path = output_left_path.strip()
                            output_right_path = output_right_path.strip()

                            mri_image = nib.load(mri_path)
                            mask_image = nib.load(mask_path)

                            mri_data = mri_image.get_fdata()
                            mask_data = mask_image.get_fdata()

                            extracted_segment_left = np.where((mask_data == label_for_left_part), mri_data, 0)
                            left_image = nib.Nifti1Image(extracted_segment_left, affine=mri_image.affine, header=mri_image.header)

                            extracted_segment_right = np.where((mask_data == label_for_right_part), mri_data, 0)
                            right_image = nib.Nifti1Image(extracted_segment_right, affine=mri_image.affine, header=mri_image.header)
                            
                            nib.save(left_image, output_left_path)
                            nib.save(right_image, output_right_path)

                            count = count + 1
                            print(mri_path , 'processed (', count, '/', num_of_files, ')')
                        except Exception as e:
                            print(f"An error occurred while processing {mri_path} and {mask_path}: {e}")
                    print('Extraction Complete')
                    
            process_images(resampled_file_path, mask_file_path, segment_left_path, segment_right_path)

            ###############################################
            # crop and pad images
            ###############################################

            max_left = crop(segment_left_path, cropped_left_path, num_of_files = num_of_files)
            max_right = crop(segment_right_path, cropped_right_path, num_of_files = num_of_files)
            max_both = [max(max_left[i], max_right[i]) for i in range(3)]
            print('Max dimensions = ', max_both)
            pad(cropped_left_path, padded_left_path, max_both, num_of_files = num_of_files)
            pad(cropped_right_path, padded_right_path, max_both, num_of_files = num_of_files)

            ###############################################
            # save images as tensors
            ###############################################

            tensor_stack = process_nifti_files(padded_left_path, padded_right_path)
            save_path = tensordir + part_to_segment.lower() + '_dataset.pt'
            torch.save(tensor_stack, save_path)
            print(f'All images saved as tensors at {save_path}')
            print(f'Processing of {part_to_segment} Complete!')

        elif label_for_left_part == label_for_right_part :
            part_to_segment = part_to_segment.replace('-', '_')
            label_for_part = label_for_left_part
            # Dirs to create
            segment_path = srcdir + '/' + 'nifti_'+ part_to_segment.lower() + '_paths.txt'
            cropped_path = srcdir + '/' + 'nifti_'+ part_to_segment.lower() + '_cropped_paths.txt'
            padded_path = srcdir + '/' + 'nifti_'+ part_to_segment.lower() + '_padded_paths.txt'

            ###############################################
            # OS filepaths & txt files configurations
            ###############################################
            
            tensordir = createdirs_NLR(mask_file_path, outputdir, part_to_segment, segment_path, cropped_path, padded_path)
        
            ###############################################
            # Extract Images
            ###############################################

            def process_images_nLR(mri_file, mask_file, output_left_file):
                with open(mri_file, 'r') as mri_paths, open(mask_file, 'r') as mask_paths, open(output_left_file, 'r') as output_left:
                    print(f'Starting Extraction of Part : {part_to_segment}...')
                    count = 0
                    for mri_path, mask_path, output_left_path in zip(mri_paths, mask_paths, output_left):
                        try:
                            mri_path = mri_path.strip()
                            mask_path = mask_path.strip()
                            output_left_path = output_left_path.strip()

                            mri_image = nib.load(mri_path)
                            mask_image = nib.load(mask_path)

                            mri_data = mri_image.get_fdata()
                            mask_data = mask_image.get_fdata()

                            extracted_segment_left = np.where((mask_data == label_for_part), mri_data, 0)
                            left_image = nib.Nifti1Image(extracted_segment_left, affine=mri_image.affine, header=mri_image.header)
                            
                            nib.save(left_image, output_left_path)

                            count = count + 1
                            print(mri_path , 'processed (', count, '/', num_of_files, ')')
                        except Exception as e:
                            print(f"An error occurred while processing {mri_path} and {mask_path}: {e}")
                    print('Extraction Complete')
                    
            process_images_nLR(resampled_file_path, mask_file_path, segment_path)

            ###############################################
            # crop and pad images
            ###############################################

            max1 = crop(segment_path, cropped_path, num_of_files = num_of_files)
            print('Max dimensions = ', max1)
            pad(cropped_path, padded_path, max1, num_of_files = num_of_files)

            ###############################################
            # save images as tensors
            ###############################################

            tensor_stack = process_nifti_files_nLR(padded_path)
            save_path = tensordir + part_to_segment.lower() + '_dataset.pt'
            torch.save(tensor_stack, save_path)
            print(f'All images saved as tensors at {save_path}')
            print(f'Processing of {part_to_segment} Complete!')

    print('Processing Complete!')
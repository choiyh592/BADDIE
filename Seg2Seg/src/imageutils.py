import numpy as np
import nibabel as nib
import torch

def find_bounding_box(data):
    """
    This function finds the bounding box of the non-zero elements in the data.
    Args:
    data: np.ndarray: 3D array of the data
    Returns:
    x_min: int: minimum x coordinate
    x_max: int: maximum x coordinate
    y_min: int: minimum y coordinate
    y_max: int: maximum y coordinate
    z_min: int: minimum z coordinate
    z_max: int: maximum z coordinate
    """
    non_zero_coords = np.argwhere(data)
    
    if non_zero_coords.size > 0:
        x_min, y_min, z_min = non_zero_coords.min(axis=0)
        x_max, y_max, z_max = non_zero_coords.max(axis=0)
    else:
        x_min = y_min = z_min = 0
        x_max = y_max = z_max = 1

    return x_min, x_max, y_min, y_max, z_min, z_max


def find_max_dimensions(file_paths):
    """
    This function finds the maximum dimensions of the nifti files in the file_paths
    Args:
    file_paths: list: list of file paths
    Returns:
    max_dims: list: list of maximum dimensions
    """

    max_dims = [0, 0, 0]
    for file_path in file_paths:
        nifti_image = nib.load(file_path)
        data = nifti_image.get_fdata()
        max_dims = [max(max_dims[i], data.shape[i]) for i in range(3)]
    return max_dims

def crop_and_save_nifti(input_path, output_path):
    """
    This function crops the nifti file and saves it to the output path
    Args:
    input_path: str: input file path
    output_path: str: output file path
    Returns:
    cropped_size: tuple: size of the cropped data
    """

    nifti_image = nib.load(input_path)
    data = nifti_image.get_fdata()

    x_min, x_max, y_min, y_max, z_min, z_max = find_bounding_box(data)
    cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    new_image = nib.Nifti1Image(cropped_data, affine=nifti_image.affine)
    nib.save(new_image, output_path)
    cropped_size = cropped_data.shape
    return cropped_size

def pad_to_center(data, target_shape=[64, 64, 64]):
    """
    This function pads the data to the target shape
    Args:
    data: np.ndarray: 3D array of the data
    target_shape: list: target shape
    Returns:
    padded_data: np.ndarray: 3D array of the padded data
    """

    pad_widths = []
    for d, t in zip(data.shape, target_shape):
        padding = (t - d) // 2
        pad_widths.append((padding, t - d - padding))
    
    return np.pad(data, pad_widths, mode='constant')

def crop(input_folder, cropped_folder, num_of_files = 0):
    """
    This function crops the nifti files in the input folder and saves them to the cropped folder
    Args:
    input_folder: str: input folder path
    cropped_folder: str: cropped folder path
    num_of_files: int: number of files
    Returns:
    max_dims: list: list of maximum dimensions
    """

    file_paths = []
    cropped_file_paths = []
    max_dims = [0, 0, 0]

    with open(input_folder, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_paths.append(line.strip())

    with open(cropped_folder, 'r') as file:
        lines = file.readlines()
        for line in lines:
            cropped_file_paths.append(line.strip())

    for i in range(len(file_paths)):
        file_name = file_paths[i]
        cropped_nifti_path = cropped_file_paths[i]
        cropped_shape = crop_and_save_nifti(file_name, cropped_nifti_path)
        max_dims = [max(max_dims[i], cropped_shape[i]) for i in range(3)]
        print('Cropped image saved to :', cropped_nifti_path, '(', i + 1, '/', num_of_files, ')')
    
    return max_dims

def pad(cropped_folder, padded_folder, max_dims = [64, 64, 64], num_of_files = 0):
    """
    This function pads the nifti files in the cropped folder and saves them to the padded folder
    Args:
    cropped_folder: str: cropped folder path
    padded_folder: str: padded folder path
    max_dims: list: list of maximum dimensions
    num_of_files: int: number of files
    """

    cropped_file_paths = []
    padded_nifti_paths = []

    with open(cropped_folder, 'r') as file:
        lines = file.readlines()
        for line in lines:
            cropped_file_paths.append(line.strip())

    with open(padded_folder, 'r') as file:
        lines = file.readlines()
        for line in lines:
            padded_nifti_paths.append(line.strip())
    
    for i in range(len(cropped_file_paths)):
        nifti_image = nib.load(cropped_file_paths[i])
        data = nifti_image.get_fdata()
        centered_data = pad_to_center(data, max_dims)
        padded_nifti_path = padded_nifti_paths[i]
        new_image = nib.Nifti1Image(centered_data, affine=nifti_image.affine)
        nib.save(new_image, padded_nifti_path)
        print('Padded image saved to :', padded_nifti_path, '(', i + 1, '/', num_of_files, ')')

def load_nifti_file(file_path):
    """
    This function loads the nifti file and returns the data as a tensor
    Args:
    file_path: str: file path
    Returns:
    data: torch.Tensor: tensor of the data
    """

    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    data = data[np.newaxis, ...]
    return data

def process_nifti_files(text_file1_path, text_file2_path):
    """
    This function processes the nifti files in the text files and returns the data as a tensor
    Args:
    text_file1_path: str: text file path
    text_file2_path: str: text file path
    Returns:
    tensors1: torch.Tensor: tensor of the data
    """

    with open(text_file1_path, 'r') as file:
        file_paths1 = file.readlines()
    tensors1 = [torch.as_tensor(load_nifti_file(path.strip()), dtype=torch.float32) for path in file_paths1]
    with open(text_file2_path, 'r') as file:
        file_paths2 = file.readlines()
    tensors2 = [torch.as_tensor(load_nifti_file(path.strip()), dtype=torch.float32) for path in file_paths2]
    tensors1.extend(tensors2)
    print('Processing Complete!')
    return torch.stack(tensors1)

def process_nifti_files_nLR(text_file1_path):
    """
    This function processes the nifti files in the text file and returns the data as a tensor
    Args:
    text_file1_path: str: text file path
    Returns:
    tensors1: torch.Tensor: tensor of the data
    """

    with open(text_file1_path, 'r') as file:
        file_paths1 = file.readlines()
    tensors1 = [torch.as_tensor(load_nifti_file(path.strip()), dtype=torch.float32) for path in file_paths1]
    print('Processing Complete!')
    return torch.stack(tensors1)

def quality_check(txt_file_path):
    """
    This function checks the quality of the nifti files in the text file
    Args:
    txt_file_path: str: text file path
    Returns:
    failed_qc_list: list: list of failed qc files
    """

    failed_qc_list = []
    with open(txt_file_path, 'r') as file:
        file_paths = file.readlines()
        for file_path in file_paths:
            nifti_image = nib.load(file_path.strip())
            data = nifti_image.get_fdata()
            if data.shape == (2, 2, 2):
                failed_qc_list.append(file_path.strip())
        
        return failed_qc_list

def calculate_volume(nifti_path, mask_path, roi_value):
    # Load the NIfTI image and the segmentation mask
    nifti_img = nib.load(nifti_path)
    mask_img = nib.load(mask_path)

    # Extract the data arrays from the images
    mask_data = mask_img.get_fdata()

    # Define the voxel dimensions (assumes isotropic voxels)
    voxel_size = np.prod(nifti_img.header.get_zooms())

    # Count the number of voxels within the specified ROI
    roi_voxel_count = np.sum(mask_data == roi_value)

    # Calculate the volume by multiplying voxel count by voxel size
    roi_volume = roi_voxel_count * voxel_size

    return roi_volume
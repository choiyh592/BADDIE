import numpy as np
import nibabel as nib
import torch

def find_bounding_box(data):
    non_zero_coords = np.argwhere(data)
    
    if non_zero_coords.size > 0:
        x_min, y_min, z_min = non_zero_coords.min(axis=0)
        x_max, y_max, z_max = non_zero_coords.max(axis=0)
    else:
        x_min = y_min = z_min = 0
        x_max = y_max = z_max = 1

    return x_min, x_max, y_min, y_max, z_min, z_max


def find_max_dimensions(file_paths):
    max_dims = [0, 0, 0]
    for file_path in file_paths:
        nifti_image = nib.load(file_path)
        data = nifti_image.get_fdata()
        max_dims = [max(max_dims[i], data.shape[i]) for i in range(3)]
    return max_dims

def crop_and_save_nifti(input_path, output_path):
    nifti_image = nib.load(input_path)
    data = nifti_image.get_fdata()

    x_min, x_max, y_min, y_max, z_min, z_max = find_bounding_box(data)
    cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    new_image = nib.Nifti1Image(cropped_data, affine=nifti_image.affine)
    nib.save(new_image, output_path)
    cropped_size = cropped_data.shape
    return cropped_size

def pad_to_center(data, target_shape=[64, 64, 64]):
    pad_widths = []
    for d, t in zip(data.shape, target_shape):
        padding = (t - d) // 2
        pad_widths.append((padding, t - d - padding))
    
    return np.pad(data, pad_widths, mode='constant')

def crop(input_folder, cropped_folder, num_of_files = 0):
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
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()
            data = data[np.newaxis, ...]
            return data

def process_nifti_files(text_file1_path, text_file2_path):
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
    with open(text_file1_path, 'r') as file:
        file_paths1 = file.readlines()
    tensors1 = [torch.as_tensor(load_nifti_file(path.strip()), dtype=torch.float32) for path in file_paths1]
    print('Processing Complete!')
    return torch.stack(tensors1)
"""
This file contains all the utilities used in that project. They are classified in 5 categories:
1- loading/saving functions:
    -load_volume
    -save_volume
    -get_volume_info
    -load_array_if_path
    -write_pickle
    -read_pickle
2- reformatting functions
    -reformat_to_list
    -reformat_to_n_channels_array
3- path related functions
    -list_images_in_folder
    -list_files
    -list_subfolders
    -strip_extension
    -strip_suffix
    -mkdir
    -mkcmd
4- shape-related functions
    -get_dims
    -get_resample_shape
    -add_axis
    -get_padding_margin
5- build affine matrices/tensors
    -create_affine_transformation_matrix
    -sample_affine_transform
    -create_rotation_transform
    -create_shearing_transform
6- miscellaneous
    -LoopInfo
    -find_closest_number_divisible_by_m


If you use this code, please cite the first SynthSeg paper:
https://github.com/BBillot/lab2im/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


import os
import glob
import math
import time
import pickle
import numpy as np
import nibabel as nib
from datetime import timedelta

# ---------------------------------------------- loading/saving functions ----------------------------------------------

def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with an identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        from lab2im import edit_volumes  # the import is done here to avoid import loops
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = edit_volumes.align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """

    mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty = nib.Nifti1Image(volume, aff, header)
            nifty.set_data_dtype(dtype)
        else:
            nifty = nib.Nifti1Image(volume, aff, header)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def get_volume_info(path_volume, return_volume=False, aff_ref=None, max_channels=10):
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels, header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :param max_channels: maximum possible number of channels for the input volume.
    :return: volume (if return_volume is true), and corresponding info. If aff_ref is not None, the returned aff is
    the original one, i.e. the affine of the image before being aligned to aff_ref.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if '.nii' in path_volume:
        data_res = np.array(header['pixdim'][1:n_dims + 1])
    elif '.mgz' in path_volume:
        data_res = np.array(header['delta'])  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        from lab2im import edit_volumes  # the import is done here to avoid import loops
        ras_axes = edit_volumes.get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = edit_volumes.get_ras_axes(aff_ref, n_dims=n_dims)
        im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var


def write_pickle(filepath, obj):
    """ write a python object with a pickle at a given path"""
    with open(filepath, 'wb') as file:
        pickler = pickle.Pickler(file)
        pickler.dump(obj)


def read_pickle(filepath):
    """ read a python object with a pickle"""
    with open(filepath, 'rb') as file:
        unpickler = pickle.Unpickler(file)
        return unpickler.load()


# ----------------------------------------------- reformatting functions -----------------------------------------------


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this function returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformatted list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int32, np.int64, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var


def reformat_to_n_channels_array(var, n_dims=3, n_channels=1):
    """This function takes an int, float, list or tuple and reformat it to an array of shape (n_channels, n_dims).
    If resolution is a str, it will be assumed to be the path of a numpy array.
    If resolution is a numpy array, it will be checked to have shape (n_channels, n_dims).
    Finally if resolution is None, this function returns None as well."""
    if var is None:
        return [None] * n_channels
    if isinstance(var, str):
        var = np.load(var)
    # convert to numpy array
    if isinstance(var, (int, float, list, tuple)):
        var = reformat_to_list(var, n_dims)
        var = np.tile(np.array(var), (n_channels, 1))
    # check shape if numpy array
    elif isinstance(var, np.ndarray):
        if n_channels == 1:
            var = var.reshape((1, n_dims))
        else:
            if np.squeeze(var).shape == (n_dims,):
                var = np.tile(var.reshape((1, n_dims)), (n_channels, 1))
            elif var.shape != (n_channels, n_dims):
                raise ValueError('if array, var should be {0} or {1}'.format((1, n_dims), (n_channels, n_dims)))
    else:
        raise TypeError('var should be int, float, list, tuple or ndarray')
    return np.round(var, 3)


# ----------------------------------------------- path-related functions -----------------------------------------------


def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz within a folder."""
    basename = os.path.basename(path_dir)
    if include_single_image & \
            (('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, 'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images


def list_files(path_dir, whole_path=True, expr=None, cond_type='or'):
    """This function returns a list of files contained in a folder, with possible regexp.
    :param path_dir: path of a folder
    :param whole_path: (optional) whether to return whole path or just the filenames.
    :param expr: (optional) regexp for files to list. Can be a str or a list of str.
    :param cond_type: (optional) if exp is a list, specify the logical link between expressions in exp.
    Can be 'or', or 'and'.
    :return: a list of files
    """
    assert isinstance(whole_path, bool), "whole_path should be bool"
    assert cond_type in ['or', 'and'], "cond_type should be either 'or', or 'and'"
    if whole_path:
        files_list = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir)
                             if os.path.isfile(os.path.join(path_dir, f))])
    else:
        files_list = sorted([f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))])
    if expr is not None:  # assumed to be either str or list of str
        if isinstance(expr, str):
            expr = [expr]
        elif not isinstance(expr, (list, tuple)):
            raise Exception("if specified, 'expr' should be a string or list of strings.")
        matched_list_files = list()
        for match in expr:
            tmp_matched_files_list = sorted([f for f in files_list if match in os.path.basename(f)])
            if cond_type == 'or':
                files_list = [f for f in files_list if f not in tmp_matched_files_list]
                matched_list_files += tmp_matched_files_list
            elif cond_type == 'and':
                files_list = tmp_matched_files_list
                matched_list_files = tmp_matched_files_list
        files_list = sorted(matched_list_files)
    return files_list


def list_subfolders(path_dir, whole_path=True, expr=None, cond_type='or'):
    """This function returns a list of subfolders contained in a folder, with possible regexp.
    :param path_dir: path of a folder
    :param whole_path: (optional) whether to return whole path or just the subfolder names.
    :param expr: (optional) regexp for files to list. Can be a str or a list of str.
    :param cond_type: (optional) if exp is a list, specify the logical link between expressions in exp.
    Can be 'or', or 'and'.
    :return: a list of subfolders
    """
    assert isinstance(whole_path, bool), "whole_path should be bool"
    assert cond_type in ['or', 'and'], "cond_type should be either 'or', or 'and'"
    if whole_path:
        subdirs_list = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir)
                               if os.path.isdir(os.path.join(path_dir, f))])
    else:
        subdirs_list = sorted([f for f in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, f))])
    if expr is not None:  # assumed to be either str or list of str
        if isinstance(expr, str):
            expr = [expr]
        elif not isinstance(expr, (list, tuple)):
            raise Exception("if specified, 'expr' should be a string or list of strings.")
        matched_list_subdirs = list()
        for match in expr:
            tmp_matched_list_subdirs = sorted([f for f in subdirs_list if match in os.path.basename(f)])
            if cond_type == 'or':
                subdirs_list = [f for f in subdirs_list if f not in tmp_matched_list_subdirs]
                matched_list_subdirs += tmp_matched_list_subdirs
            elif cond_type == 'and':
                subdirs_list = tmp_matched_list_subdirs
                matched_list_subdirs = tmp_matched_list_subdirs
        subdirs_list = sorted(matched_list_subdirs)
    return subdirs_list


def get_image_extension(path):
    name = os.path.basename(path)
    if name[-7:] == '.nii.gz':
        return 'nii.gz'
    elif name[-4:] == '.mgz':
        return 'mgz'
    elif name[-4:] == '.nii':
        return 'nii'
    elif name[-4:] == '.npz':
        return 'npz'


def strip_extension(path):
    """Strip classical image extensions (.nii.gz, .nii, .mgz, .npz) from a filename."""
    return path.replace('.nii.gz', '').replace('.nii', '').replace('.mgz', '').replace('.npz', '')


def strip_suffix(path):
    """Strip classical image suffix from a filename."""
    path = path.replace('_aseg', '')
    path = path.replace('aseg', '')
    path = path.replace('.aseg', '')
    path = path.replace('_aseg_1', '')
    path = path.replace('_aseg_2', '')
    path = path.replace('aseg_1_', '')
    path = path.replace('aseg_2_', '')
    path = path.replace('_orig', '')
    path = path.replace('orig', '')
    path = path.replace('.orig', '')
    path = path.replace('_norm', '')
    path = path.replace('norm', '')
    path = path.replace('.norm', '')
    path = path.replace('_talairach', '')
    path = path.replace('GSP_FS_4p5', 'GSP')
    path = path.replace('.nii_crispSegmentation', '')
    path = path.replace('_crispSegmentation', '')
    path = path.replace('_seg', '')
    path = path.replace('.seg', '')
    path = path.replace('seg', '')
    path = path.replace('_seg_1', '')
    path = path.replace('_seg_2', '')
    path = path.replace('seg_1_', '')
    path = path.replace('seg_2_', '')
    return path


def mkdir(path_dir):
    """Recursively creates the current dir as well as its parent folders if they do not already exist."""
    if path_dir[-1] == '/':
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)


def mkcmd(*args):
    """Creates terminal command with provided inputs.
    Example: mkcmd('mv', 'source', 'dest') will give 'mv source dest'."""
    return ' '.join([str(arg) for arg in args])


# ---------------------------------------------- shape-related functions -----------------------------------------------


def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def get_resample_shape(patch_shape, factor, n_channels=None):
    """Compute the shape of a resampled array given a shape factor.
    :param patch_shape: size of the initial array (without number of channels).
    :param factor: resampling factor. Can be a number, sequence, or 1d numpy array.
    :param n_channels: (optional) if not None, add a number of channel at the end of the computed shape.
    :return: list containing the shape of the input array after being resampled by the given factor.
    """
    factor = reformat_to_list(factor, length=len(patch_shape))
    shape = [math.ceil(patch_shape[i] * factor[i]) for i in range(len(patch_shape))]
    if n_channels is not None:
        shape += [n_channels]
    return shape


def add_axis(x, axis=0):
    """Add axis to a numpy array.
    :param x: input array
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x


def get_padding_margin(cropping, loss_cropping):
    """Compute padding margin"""
    if (cropping is not None) & (loss_cropping is not None):
        cropping = reformat_to_list(cropping)
        loss_cropping = reformat_to_list(loss_cropping)
        n_dims = max(len(cropping), len(loss_cropping))
        cropping = reformat_to_list(cropping, length=n_dims)
        loss_cropping = reformat_to_list(loss_cropping, length=n_dims)
        padding_margin = [int((cropping[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        if len(padding_margin) == 1:
            padding_margin = padding_margin[0]
    else:
        padding_margin = None
    return padding_margin

# --------------------------------------------------- miscellaneous ----------------------------------------------------
class LoopInfo:
    """
    Class to print the current iteration in a for loop, and optionally the estimated remaining time.
    Instantiate just before the loop, and call the update method at the start of the loop.
    The printed text has the following format:
    processing i/total    remaining time: hh:mm:ss
    """

    def __init__(self, n_iterations, spacing=10, text='processing', print_time=False):
        """
        :param n_iterations: total number of iterations of the for loop.
        :param spacing: frequency at which the update info will be printed on screen.
        :param text: text to print. Default is processing.
        :param print_time: whether to print the estimated remaining time. Default is False.
        """

        # loop parameters
        self.n_iterations = n_iterations
        self.spacing = spacing

        # text parameters
        self.text = text
        self.print_time = print_time
        self.print_previous_time = False
        self.align = len(str(self.n_iterations)) * 2 + 1 + 3

        # timing parameters
        self.iteration_durations = np.zeros((n_iterations,))
        self.start = time.time()
        self.previous = time.time()

    def update(self, idx):

        # time iteration
        now = time.time()
        self.iteration_durations[idx] = now - self.previous
        self.previous = now

        # print text
        if idx == 0:
            print(self.text + ' 1/{}'.format(self.n_iterations))
        elif idx % self.spacing == self.spacing - 1:
            iteration = str(idx + 1) + '/' + str(self.n_iterations)
            if self.print_time:
                # estimate remaining time
                max_duration = np.max(self.iteration_durations)
                average_duration = np.mean(self.iteration_durations[self.iteration_durations > .01 * max_duration])
                remaining_time = int(average_duration * (self.n_iterations - idx))
                # print total remaining time only if it is greater than 1s or if it was previously printed
                if (remaining_time > 1) | self.print_previous_time:
                    eta = str(timedelta(seconds=remaining_time))
                    print(self.text + ' {:<{x}} remaining time: {}'.format(iteration, eta, x=self.align))
                    self.print_previous_time = True
                else:
                    print(self.text + ' {}'.format(iteration))
            else:
                print(self.text + ' {}'.format(iteration))

def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher' (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)

    # first = f(0), last = f(+inf), fix_point = [x0, f(x0))]
    a = last
    b = first - last
    c = - (1 / fix_point[0]) * np.log((fix_point[1] - last) / (first - last))
    return a + b * np.exp(-c * x)
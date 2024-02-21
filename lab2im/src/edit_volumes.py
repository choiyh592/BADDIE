"""
This file contains functions to edit/preprocess volumes (i.e. not tensors!).
These functions are sorted in five categories:
1- volume editing: this can be applied to any volume (i.e. images or label maps). It contains:
        -rescale_volume
        -crop_volume
        -crop_volume_with_idx
        -pad_volume
        -flip_volume
        -resample_volume
        -get_ras_axes
        -align_volume_to_ref

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


# python imports
import os
import csv
import shutil
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import label as scipy_label
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

# project imports
from lab2im.src import utils
# ---------------------------------------------------- edit volume -----------------------------------------------------

def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2, max_percentile=98, use_positive_only=False):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)


def crop_volume(volume, cropping_margin=None, cropping_shape=None, aff=None, return_crop_idx=False, mode='center'):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. The cropping margin is applied on both sides.
    Can be an int, sequence or 1d numpy array of size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :param return_crop_idx: (optional) whether to return the cropping indices used to crop the given volume.
    :param mode: (optional) if cropping_shape is not None, whether to extract the centre of the image (mode='center'),
    or to randomly crop the volume to the provided shape (mode='random'). Default is 'center'.
    :return: cropped volume, corresponding affine matrix if aff is not None, and cropping indices if return_crop_idx is
    True (in that order).
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = utils.get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = utils.reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i] for i in range(n_dims)]
    else:
        cropping_shape = utils.reformat_to_list(cropping_shape, length=n_dims)
        if mode == 'center':
            min_crop_idx = np.maximum([int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum([min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                                      np.array(vol_shape)[:n_dims])
        elif mode == 'random':
            crop_max_val = np.maximum(np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    # sort outputs
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)

def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=None, return_copy=True):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: cropping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if aff is specified, this function returns an updated affine matrix of the volume after
    cropping.
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims

    # crop image
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return new_volume, aff
    else:
        return new_volume


def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :param return_pad_idx: (optional) the pad_idx corresponds to the indices where we should crop the resulting
    padded image (ie the output of this function) to go back to the original volume (ie the input of this function).
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = utils.get_dims(vol_shape)
    padding_shape = utils.reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # check if need to pad
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):

        # get padding margins
        min_margins = np.maximum(np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)


def flip_volume(volume, axis=None, direction=None, aff=None, return_copy=True):
    """Flip volume along a specified axis.
    If unknown, this axis can be inferred from an affine matrix with a specified anatomical direction.
    :param volume: a numpy array
    :param axis: (optional) axis along which to flip the volume. Can either be an int or a tuple.
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' anterior/posterior), 'si' (superior/inferior).
    :param aff: (optional) please provide an affine matrix if direction is not None
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: flipped volume
    """

    new_volume = volume.copy() if return_copy else volume
    assert (axis is not None) | ((aff is not None) & (direction is not None)), \
        'please provide either axis, or an affine matrix with a direction'

    # get flipping axis from aff if axis not provided
    if (axis is None) & (aff is not None):
        volume_axes = get_ras_axes(aff)
        if direction == 'rl':
            axis = volume_axes[0]
        elif direction == 'ap':
            axis = volume_axes[1]
        elif direction == 'si':
            axis = volume_axes[2]
        else:
            raise ValueError("direction should be 'rl', 'ap', or 'si', had %s" % direction)

    # flip volume
    return np.flip(new_volume, axis=axis)


def resample_volume(volume, aff, new_vox_size, interpolation='linear', blur=True):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :param interpolation: (optional) type of interpolation. Can be 'linear' or 'nearest'. Default is 'linear'.
    :param blur: (optional) whether to blur before resampling to avoid aliasing effects.
    Only used if the input volume is downsampled. Default is True.
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas) if blur else volume

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = RegularGridInterpolator((x, y, z), volume_filt, method=interpolation)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of length n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = utils.get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume



    """Check if all volumes within the same folder share the same characteristics: shape, affine matrix, resolution.
    Also have option to check if all volumes have the same intensity values (useful for label maps).
    :return four lists, each containing the different values detected for a specific parameter among those to check."""

    # define information to check
    list_shape = list()
    list_aff = list()
    list_res = list()
    list_axes = list()
    if check_values:
        list_unique_values = list()
    else:
        list_unique_values = None

    # loop through files
    path_images = utils.list_images_in_folder(image_dir)
    loop_info = utils.LoopInfo(len(path_images), 10, 'checking', verbose) if verbose else None
    for idx, path_image in enumerate(path_images):
        if loop_info is not None:
            loop_info.update(idx)

        # get info
        im, shape, aff, n_dims, _, h, res = utils.get_volume_info(path_image, True, np.eye(4), max_channels)
        axes = get_ras_axes(aff, n_dims=n_dims).tolist()
        aff[:, np.arange(n_dims)] = aff[:, axes]
        aff = (np.int32(np.round(np.array(aff[:3, :3]), 2) * 100) / 100).tolist()
        res = (np.int32(np.round(np.array(res), 2) * 100) / 100).tolist()

        # add values to list if not already there
        if (shape not in list_shape) | (not keep_unique):
            list_shape.append(shape)
        if (aff not in list_aff) | (not keep_unique):
            list_aff.append(aff)
        if (res not in list_res) | (not keep_unique):
            list_res.append(res)
        if (axes not in list_axes) | (not keep_unique):
            list_axes.append(axes)
        if list_unique_values is not None:
            uni = np.unique(im).tolist()
            if (uni not in list_unique_values) | (not keep_unique):
                list_unique_values.append(uni)

    return list_shape, list_aff, list_res, list_axes, list_unique_values

    """This function masks all label maps in a folder by keeping a set of given label values.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where corrected label maps will be writen
    :param values_to_keep: list of values for masking the label maps.
    :param masking_value: (optional) value to mask the label maps with
    :param mask_result_dir: (optional) path of directory where applied masks will be writen
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)
    if mask_result_dir is not None:
        utils.mkdir(mask_result_dir)

    # reformat values to keep
    values_to_keep = utils.reformat_to_list(values_to_keep, load_as_numpy=True)

    # loop over labels
    path_labels = utils.list_images_in_folder(labels_dir)
    loop_info = utils.LoopInfo(len(path_labels), 10, 'masking', True)
    for idx, path_label in enumerate(path_labels):
        loop_info.update(idx)

        # mask labels
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if mask_result_dir is not None:
            path_result_mask = os.path.join(mask_result_dir, os.path.basename(path_label))
        else:
            path_result_mask = ''
        if (not os.path.isfile(path_result)) | \
                (mask_result_dir is not None) & (not os.path.isfile(path_result_mask)) | \
                recompute:
            lab, aff, h = utils.load_volume(path_label, im_only=False)
            if mask_result_dir is not None:
                labels, mask = mask_label_map(lab, values_to_keep, masking_value, return_mask=True)
                path_result_mask = os.path.join(mask_result_dir, os.path.basename(path_label))
                utils.save_volume(mask, aff, h, path_result_mask)
            else:
                labels = mask_label_map(lab, values_to_keep, masking_value, return_mask=False)
            utils.save_volume(labels, aff, h, path_result)
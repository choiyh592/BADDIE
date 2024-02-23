from lab2im.src import utils, edit_volumes
import numpy as np

# n_levels = ensures that the input dimensions are divisible by 2^n
def preprocess(path_image, ct, target_res=1., n_levels=5, crop=None, min_pad=None, path_resample=None):

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)
    if n_dims == 2 and 1 < n_channels < 4:
        raise Exception('either the input is 2D with several channels, or is 3D with at most 3 slices. '
                        'Either way, results are going to be poor...')
    elif n_dims == 2 and 3 < n_channels < 11:
        print('warning: input with very few slices')
        n_dims = 3
    elif n_dims < 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    elif n_dims > 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_channels > 1:
        print('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # resample image # if necessary
    target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
    # if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
    im_res = target_res
    im, aff = edit_volumes.resample_volume(im, aff, im_res)
    if path_resample is not None:
        utils.save_volume(im, aff, h, path_resample)

    return im, aff, h, im_res
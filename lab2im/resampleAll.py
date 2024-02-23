from lab2im.src.resample import preprocess


def resampleAll(path_images, path_resampled_images, cropping=None, min_pad=128):
    """Resample all images in a list to a new resolution.
    :param path_images: list of paths to the original images
    :param path_resampled_images: list of paths to the resampled images
    :param cropping: cropping parameters for the resampling
    :param min_pad: minimum padding for the resampling
    """

    listlength = len(path_images)
    for i in range(listlength):
        print(f"Resampling {path_images[i]} to {path_resampled_images[i]}... ({i+1}/{listlength})")
        _, _, _, _ = preprocess(path_image=path_images[i],
                                                ct=False,
                                                crop=cropping,
                                                min_pad=min_pad,
                                                path_resample=path_resampled_images[i])
        

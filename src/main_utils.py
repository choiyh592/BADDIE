import argparse

def read_file(file_path):
    # Open the file in read mode
    with open(file_path, "r") as file:
        lines = file.readlines()  # Read all lines into a list

    # Remove newline characters from each line and store in a new list
    lines = [line.strip() for line in lines]

    # Print the list of lines
    return lines

def create_text_files(seg_path, res_path, txt_path, sids):
        conformed_paths = []
        segmented_paths = []
        resampled_paths = []
        for sid in sids:
            conformed_path = seg_path / f'{sid}' / 'mri' / 'orig.mgz'
            segmented_path = seg_path / f'{sid}' / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'
            resampled_path = res_path / f'{sid}_resampled.nii.gz'
            conformed_paths.append(conformed_path)
            segmented_paths.append(segmented_path)
            resampled_paths.append(resampled_path)
        
        resampled_path_txt = txt_path / '_resampled_paths.txt'
        conformed_path_txt = txt_path / '_conformed_paths.txt'
        segmented_path_txt = txt_path / '_segmented_paths.txt'

        with open(resampled_path_txt, 'w') as file:
            for path in resampled_paths:
                file.write(f'{path}\n')

        with open(conformed_path_txt, 'w') as file:
            for path in conformed_paths:
                file.write(f'{path}\n')
        
        with open(segmented_path_txt, 'w') as file:
            for path in segmented_paths:
                file.write(f'{path}\n')

        return resampled_path_txt, conformed_path_txt, segmented_path_txt

# Define a function to create a custom parser
def create_parser(iteration, t1, sd, device, sid):
    """Create a parser for the current iteration.
    Note that this approach of using parsers may not adhere to best practices.
    This approach is for conveniently reusing the code in the FastSurferCNN package.
    :param iteration: the current iteration
    :param t1: the name of the T1 full head MRI
    :param sd: the directory in which evaluation results should be written
    :param device: the device to run inference on
    :param sid: the subject id to use
    Returns:
    parser: An argparse.ArgumentParser object configured with the specified options.
    """

    seg_log = f'{sd}/{sid}/seg_log.log'
    qc_log = f'{sd}/{sid}/qc_log.log'

    parser = argparse.ArgumentParser(description=f'Parser for iteration {iteration}')

    # Unused
    parser.add_argument('--inputtxt', type=str, help='Text File containing T1 MRI paths for all subjects. One path per line.')
    parser.add_argument('--outputdir', type=str, help='Path for a single directory to contain all outputs.')
    parser.add_argument('--sidtxt', type=str, help='Text File containing the subject ID to use. One ID per line.' , default=None)

    # Used
    parser.add_argument('--t1', type=str, help='Name of T1 full head MRI. Absolute path if single image else '
        "common image name. Default: mri/orig.mgz" , default=t1, dest="orig_name")
    parser.add_argument('--sd', type=str, help='Directory in which evaluation results should be written. '
        'Will be created if it does not exist. Optional if full path is defined for --pred_name.' , default=sd, dest="out_dir")
    parser.add_argument('--seg_log', type=str, help='Absolute path to file in which run logs will be saved. If not set, logs will '
        'not be saved.' , default=seg_log, dest="log_name")
    parser.add_argument('--qc_log', type=str, help='Absolute path to file in which quality control logs will be saved. If not set, logs will '
        'not be saved.' , default=qc_log, dest="qc_log")
    parser.add_argument('--device', type=str, help='Select device to run inference on: cpu, or cuda (= Nvidia gpu) or specify a certain gpu '
        '(e.g. cuda:1), default: auto' , default=device)
    parser.add_argument('--sid', type=str, help='Directly set the subject id to use. Can be used for single subject input. For multi-subject '
        'processing, use remove suffix if sid is not second to last element of input file passed to --t1' , default=sid, dest="sid")
    
    return parser
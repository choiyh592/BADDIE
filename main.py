# Description: This is the main file for the 24BrainMRI_Preprocessing package. It is used to run the entire pipeline.
import argparse
import os
import sys
import logging
from pathlib import Path

# local imports
from lab2im.resampleAll import resampleAll
from FastSurferCNN.run_prediction import run_single_prediction
from Seg2Seg.extract_all import process_images
from src.main_utils import read_file, create_parser, create_text_files

if __name__ == '__main__':
    print("Starting 24BrainMRI_Preprocessing.")
    # Parse the arguments
    parser = argparse.ArgumentParser(description='24BrainMRI_Preprocessing')
    parser.add_argument('--inputtxt', type=str, help='Text File containing absolute T1 MRI paths for all subjects. One path per line.')
    parser.add_argument('--sidtxt', type=str, help='Text File containing the **Unique** subject ID to use. One ID per line.'
                        'Should be in the same order as the --inputtxt file. Should have the same number of lines as the --inputtxt file.')
    parser.add_argument('--outputdir', type=str, help='Path for a single directory to contain all outputs.')
    parser.add_argument('--device', type=str, help='Select device to run FastSurferCNN inference on: cpu, or cuda(= Nvidia GPU) or specify a certain GPU'
        '(e.g. cuda:1), default: auto' , default='auto')

    args = parser.parse_args()

    # Create the output directory
    output_path = Path(args.outputdir)
    os.makedirs(output_path, exist_ok=True)

    # Create the resampled directory
    res_path = output_path / 'resampled'
    os.makedirs(res_path, exist_ok=True)

    # Create the segmentation directory
    seg_path = output_path / 'segmentations'
    os.makedirs(seg_path, exist_ok=True)

    # Create directories for text files
    txt_path = output_path / 'text_files'
    os.makedirs(txt_path, exist_ok=True)

    # Create directories for extracted files
    ext_path = output_path / 'extractions'
    os.makedirs(ext_path, exist_ok=True)

    # Read the text file containing the input paths
    input_file = Path(args.inputtxt)
    input_paths = read_file(input_file)

    # Read the text file containing the subject IDs
    sid_file = Path(args.sidtxt)
    sids = read_file(sid_file)

    # Create text files for the conformed and segmented paths
    resampled_path_txt, conformed_path_txt, segmented_path_txt = create_text_files(seg_path, res_path, txt_path, sids)

    # Read the created text file containing the resampled paths
    resampled_paths = read_file(resampled_path_txt)

    # Sanity check : There should be an equal number of input paths and subject IDs
    num_inputs = len(input_paths)
    try:
        assert num_inputs == len(sids), "The number of input paths and subject IDs should be equal."
    except AssertionError as e:
        logging.error(e)
        sys.exit(1)

    # Sanity check : The sids should be unique
    try:
        assert len(sids) == len(set(sids)), "The subject IDs should be unique."
    except AssertionError as e:
        logging.error(e)
        sys.exit(1)

    # Set the device
    device = args.device

    # Resample the input files using lab2im : check the README for further details
    logging.info('Running resampling of the input files using lab2im...')
    resampleAll(input_paths, resampled_paths)

    # Segment all the resampled files using FastSurferCNN : check the README for further details
    logging.info('Running segmentation of the resampled files using FastSurferCNN...')
    for i, (resampled_path, sid) in enumerate(zip(resampled_paths, sids)):
        # Create a parser for the current iteration
        new_parser = create_parser(i, resampled_path, str(seg_path), device=device, sid=sid)

        # Create a directory for the current subject
        os.makedirs(seg_path / sid, exist_ok=True)

        # Run the resampling process(prediction)
        run_single_prediction(new_parser)
    
    # Extract the segmented files using Seg2Seg : check the README for further details
    logging.info('Running extraction of the segmented files using Seg2Seg...')
    process_images(conformed_path_txt, segmented_path_txt, txt_path, ext_path, num_of_files = num_inputs)
    
    logging.info("24BrainMRI_Preprocessing has finished running!")

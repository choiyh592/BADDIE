import argparse
from Seg2Seg.segment_qc import remove_qcfailed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove the failed QC filepaths from the text files')
    parser.add_argument('--outputdir', type=str, help='Path for a single directory to contain all outputs.')
    args = parser.parse_args()
    outputdir = args.outputdir

    remove_qcfailed(outputdir)
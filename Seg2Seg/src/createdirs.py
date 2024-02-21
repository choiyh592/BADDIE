import os

def createdirs_LR(mask_file_path, outputdir, part_to_segment, segment_left_path, segment_right_path, cropped_left_path,
                              cropped_right_path, padded_left_path, padded_right_path):
                
    with open(mask_file_path, 'r') as file:
        lines = file.readlines()

    dir1 = outputdir + '/' + part_to_segment.upper() + '_EXTRACT'
    if not os.path.exists(dir1):
        os.mkdir(dir1)

    dirleft = outputdir + '/' + part_to_segment.upper() + '_EXTRACT/left'
    if not os.path.exists(dirleft):
        os.mkdir(dirleft)

    dirright = outputdir + '/' + part_to_segment.upper() + '_EXTRACT/right'
    if not os.path.exists(dirright):
        os.mkdir(dirright)

    dir2 = outputdir + '/' + part_to_segment.upper() + '_CROPPED'
    if not os.path.exists(dir2):
        os.mkdir(dir2)

    cropleft = outputdir + '/' + part_to_segment.upper() + '_CROPPED/left'
    if not os.path.exists(cropleft):
        os.mkdir(cropleft)

    cropright = outputdir + '/' + part_to_segment.upper() + '_CROPPED/right'
    if not os.path.exists(cropright):
        os.mkdir(cropright)

    dir3 = outputdir + '/' + part_to_segment.upper() + '_PADDED'
    if not os.path.exists(dir3):
        os.mkdir(dir3)

    padleft = outputdir + '/' + part_to_segment.upper() + '_PADDED/left'
    if not os.path.exists(padleft):
        os.mkdir(padleft)

    padright = outputdir + '/' + part_to_segment.upper() + '_PADDED/right'
    if not os.path.exists(padright):
        os.mkdir(padright)

    tensordir = outputdir + '/' + part_to_segment.upper() + '_TENSORS/'
    if not os.path.exists(tensordir):
        os.mkdir(tensordir)

    with open(segment_left_path, 'w') as file:
        for line in lines:
            new_line = dirleft + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'_left.nii.gz'
            file.write(new_line)
            file.write('\n')

    with open(segment_right_path, 'w') as file:
        for line in lines:
            new_line = dirright + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'_right.nii.gz'
            file.write(new_line)
            file.write('\n')

    with open(cropped_left_path, 'w') as file:
        for line in lines:
            new_line = cropleft + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'_left.nii.gz'
            file.write(new_line)
            file.write('\n')

    with open(cropped_right_path, 'w') as file:
        for line in lines:
            new_line = cropright + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'_right.nii.gz'
            file.write(new_line)
            file.write('\n')

    with open(padded_left_path, 'w') as file:
        for line in lines:
            new_line = padleft + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'_left.nii.gz'
            file.write(new_line)
            file.write('\n')

    with open(padded_right_path, 'w') as file:
        for line in lines:
            new_line = padright + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'_right.nii.gz'
            file.write(new_line)
            file.write('\n')
    
    return tensordir

def createdirs_NLR(mask_file_path, outputdir, part_to_segment, segment_path, cropped_path, padded_path):
    with open(mask_file_path, 'r') as file:
        lines = file.readlines()

    dir1 = outputdir + '/' + part_to_segment.upper() + '_EXTRACT'
    if not os.path.exists(dir1):
        os.mkdir(dir1)

    croppeddir = outputdir + '/' + part_to_segment.upper() + '_CROPPED'
    if not os.path.exists(croppeddir):
        os.mkdir(croppeddir)

    paddeddir = outputdir + '/' + part_to_segment.upper() + '_PADDED'
    if not os.path.exists(paddeddir):
        os.mkdir(paddeddir)

    tensordir = outputdir + '/' + part_to_segment.upper() + '_TENSORS/'
    if not os.path.exists(tensordir):
        os.mkdir(tensordir)

    with open(segment_path, 'w') as file:
        for line in lines:
            new_line = dir1 + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'.nii.gz'
            file.write(new_line)
            file.write('\n')

    with open(cropped_path, 'w') as file:
        for line in lines:
            new_line = croppeddir + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'.nii.gz'
            file.write(new_line)
            file.write('\n')

    with open(padded_path, 'w') as file:
        for line in lines:
            new_line = paddeddir + '/' + line.split('/')[-3] + '_' + part_to_segment.lower() +'.nii.gz'
            file.write(new_line)
            file.write('\n')

    return tensordir


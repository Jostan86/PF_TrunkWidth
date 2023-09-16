import json
import os
import numpy as np

# This script is used to add the img_x_positions to the ground truth data I generated for the feb 2023 data. The
# ground truth was generated with the PF itself, just intialized in the right location, and it's used for the
# refinement tests. However, it didn't have the img_x_positions, which i need now that I'm correcting the width of
# the tree based on the trunk x position in the image.

# Directory with the ground truth data
data_dir = '/media/jostan/MOAD/research_data/2023_orchard_data/pf_data/old_data/'
file_names = ['run_' + str(i) + '_gt_data.json' for i in range(1, 10)]

# Loop through each file
for file_name in file_names:

    # Load data
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)

    # get list of keys in data dict, each key is a time stamp from the original bag file
    keys = list(data.keys())

    # Loop through each saved message
    for key in keys:
        # Get the data for the current time stamp
        cur_data = data[key]
        # Data is None if it was an image but no trunk was detected
        if cur_data is None:
            continue
        # Get the keys for the current data
        sub_keys = list(cur_data.keys())
        # Check if dict has 'tree_data' key, this means it was an image with a detected trunk
        if 'tree_data' in sub_keys:
            # Add the img_x_positions to the tree_data
            data[key]['tree_data']['img_x_positions'] = []
            for i in range(len(cur_data['tree_data']['positions'])):
                x = cur_data['tree_data']['positions'][i][0]
                y = cur_data['tree_data']['positions'][i][1]
                width_image = 640
                img_x_pos = ((x * width_image) / (2 * y * np.tan(np.deg2rad(55/2)))) + width_image/2
                data[key]['tree_data']['img_x_positions'].append(img_x_pos)

    # Save the data to a new file
    new_file_path = os.path.join(data_dir, file_name[:-5] + '_x_pos.json')
    with open(new_file_path, 'w') as f:
        json.dump(data, f, indent=4)





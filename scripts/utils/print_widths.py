import json
import os
import sys
sys.path.append('../')
from env_vars import *
import os

tree_data_path = os.environ.get('MAP_DATA_PATH')

# Load the tree data dictionary
with open(tree_data_path, 'rb') as f:
    tree_data = json.load(f)

tree_num_to_find = 4

for tree in tree_data:
    if tree['tree_num'] == tree_num_to_find:
        print(tree['width_estimate'])
        break

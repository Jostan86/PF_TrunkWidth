import json

tree_data_path = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/orchard_data_analysis/data' \
                 '/2020_11_bag_data/afternoon2/tree_list_mod3.json'

# Load the tree data dictionary
with open(tree_data_path, 'rb') as f:
    tree_data = json.load(f)

tree_num_to_find = 1

for tree in tree_data:
    if tree['tree_num'] == tree_num_to_find:
        print(tree['width_estimate'])
        break

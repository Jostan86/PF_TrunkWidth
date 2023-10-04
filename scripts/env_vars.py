#!/usr/bin/env python3

# Set some environment variables for the scripts to use

import os

os.environ['MAP_DATA_PATH'] = ('/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/orchard_data_analysis/data'
                               '/2020_11_bag_data/afternoon2/tree_list_mod4.json')
# os.environ['MAP_DATA_PATH'] = '/media/jostan/portabits/sept6/tree_list_mod4.json'

os.environ['MODEL_WEIGHT_PATH'] = "/home/jostan/OneDrive/Docs/Grad_school/Research/yolo_model/best_s_500_v7.pt"
os.environ['MODEL_STARTUP_IMAGE_PATH'] = ("/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/pf_trunk_width/data"
                                          "/rt_startup_image.png")

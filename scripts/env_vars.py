#!/usr/bin/env python3

# Set some environment variables for the scripts to use

import os

pkg_dir = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/pf_trunk_width/'
os.environ['MAP_DATA_PATH'] = (pkg_dir + 'data/tree_list_mod4.json')

# os.environ['MODEL_WEIGHT_PATH'] = "/home/jostan/OneDrive/Docs/Grad_school/Research/yolo_model/best_x_500_v7.pt"
os.environ['MODEL_WEIGHT_PATH'] = (pkg_dir + "/data/best_s_500_v7.pt")
os.environ['MODEL_STARTUP_IMAGE_PATH'] = (pkg_dir + "/data/live_startup_image.png")

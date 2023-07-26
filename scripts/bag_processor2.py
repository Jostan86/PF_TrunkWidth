#!/usr/bin/env python3

from cv_bridge import CvBridge, CvBridgeError
import os
import rosbag
import rospy
from trunk_width_estimation.trunk_analyzer import TrunkAnalyzer
import json
import numpy as np
from helper_funcs import get_map_data, ParticleMapPlotter
from pf_engine import PFEngine
import cv2
import time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


# To run in pycharm add the following environment variable:
# LD_PRELOAD: Set it to /usr/lib/x86_64-linux-gnu/libstdc++.so.6

bridge = CvBridge()

bag_file_dir = "/media/jostan/MOAD/research_data/2023_orchard_data/uncompressed/synced/pcl_mod/"

# Get the bag file names and sort them alphabetically
bag_file_names = os.listdir(bag_file_dir)
bag_file_names.sort()

# Get the bag file paths, for now just keep the first 10
bag_file_paths = []
for bag_file_name in bag_file_names:
    if bag_file_name[13] == '5':
        bag_file_paths.append(bag_file_dir + bag_file_name)

# Path to the tree data dictionary
tree_data_path = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/orchard_data_analysis/data' \
                   '/2020_11_bag_data/afternoon2/tree_list_mod3.json'

classes, positions, widths = get_map_data()

# Put the map data into a dictionary
map_data = {'classes': classes, 'positions': positions, 'widths': widths}
plotter = ParticleMapPlotter(map_data)

start_pose_center = (12.7, 53.3)
start_radius = 1.5
num_particles = 1000

ask_for_start_pose = True

if ask_for_start_pose:
    ans = None
    while ans != 'y':
        ans = input("Start pose center is {}, is this correct? (y/n)".format(start_pose_center))
        if ans == 'n':
            x = float(input("Enter x: "))
            y = float(input("Enter y: "))
            start_pose_center = (x, y)

    start_radius = float(input("Enter start pose radius: "))
    num_particles = int(input("Enter number of particles: "))

# Create the particle filter engine
pf_engine = PFEngine(map_data, start_pose_center=start_pose_center, start_pose_radius=start_radius, num_particles=num_particles,)

plotter.add_particle(pf_engine.particles)

# Create the trunk analyzer
trunk_analyzer = TrunkAnalyzer()

rgb_topic = "/registered/rgb/image"
depth_topic = "/registered/depth/image"

def check_send_seg(rgb_img, depth_img, pf_engine, depth_msg, color_msg, skip_count, skip_num=1):

    if depth_msg is not None and color_msg is not None and depth_msg.header.stamp == color_msg.header.stamp:
        # if skip_count < skip_num:
        #     skip_count += 1
        #     return

        tree_positions, widths, classes, x_pos = trunk_analyzer.pf_helper(rgb_img, depth_img, show_seg=True)

        if tree_positions is None:
            return

        # Switch sign on x_pos and y_pos
        tree_positions[:, 0] = -tree_positions[:, 0]
        tree_positions[:, 1] = -tree_positions[:, 1]

        tree_data = {'positions': tree_positions, 'widths': widths, 'classes': classes, 'img_x_pos': x_pos,
                     'header': depth_msg.header}
        pf_engine.save_scan(tree_data)
        plotter.update_particles(pf_engine.particles)

        # input("Press Enter to continue...")

# Loop through the bag files
time_prev = None
for bag_file_path in bag_file_paths:

    bag_data = rosbag.Bag(bag_file_path)

    topics = ["/registered/rgb/image", "/registered/depth/image", "/odometry/filtered"]
    # Loop through the bag file messages of topic in topics
    depth_image = None
    depth_msg = None
    color_image = None
    color_msg = None
    skip_count = 0
    save_directory = "/media/jostan/MOAD/research_data/2023_orchard_data/segs/"
    save_directory += bag_file_path.split('/')[-1].split('.')[0] + '/'


    for topic, msg, t in bag_data.read_messages(topics=topics):

        if time_prev is None:
            start_time = t
            # Convert rospy.Time to seconds
            start_time = start_time.to_sec()
            time_prev = t
        if t.to_sec() - start_time < 40:
            continue

        if topic == '/throttled/camera/depth/image_rect_raw' or topic == "/registered/depth/image":
            try:
                depth_msg = msg
                depth_image = bridge.imgmsg_to_cv2(msg, "passthrough")

            except CvBridgeError as e:
                print(e)
            check_send_seg(color_image, depth_image, pf_engine, depth_msg, color_msg, skip_count)

        elif topic == '/throttled/camera/color/image_raw' or topic == "/registered/rgb/image":
            try:
                color_msg = msg
                color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                print(e)

            check_send_seg(color_image, depth_image, pf_engine, depth_msg, color_msg, skip_count)

        elif topic == "/odometry/filtered":
            pf_engine.save_odom(msg)
            # Skip plotting if there are over 300000 particles
            if pf_engine.particles.shape[0] < 300000:
                plotter.update_particles(pf_engine.particles)


        # time_change = t - time_prev
        # time_prev = t
        #
        # # Convert rospy.Duration to seconds
        # time_change = time_change.to_sec()
        # time.sleep(time_change)









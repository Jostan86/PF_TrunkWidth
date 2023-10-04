#!/usr/bin/env python3

# This script converts the point cloud data in the bag files to RGB and depth images

import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import ros_numpy
import os

# Bag file path
bag_file_dir_og = "/media/jostan/MOAD/research_data/2023_orchard_data/uncompressed/synced/original"
bag_file_dir_new = "/media/jostan/MOAD/research_data/2023_orchard_data/uncompressed/synced/pcl_mod"

# Output topics
rgb_topic = "/registered/rgb/image"
depth_topic = "/registered/depth/image"

# Initialize CvBridge
cv_bridge = CvBridge()

# Get list of bag file names
bag_file_names = os.listdir(bag_file_dir_og)

count = 0

# Loop through the bag files
for bag_file_name in bag_file_names:
    count += 1
    print("Processing bag file {} of {}".format(count, len(bag_file_names)))
    bag_file_path = bag_file_dir_og + "/" + bag_file_name
    new_bag_file_path = bag_file_dir_new + "/" + bag_file_name.split(".")[0] + "_pcl-mod.bag"
    bag_og = rosbag.Bag(bag_file_path, "r")
    bag_out = rosbag.Bag(new_bag_file_path, "w")  # Create a temporary bag file for filtered messages

    # Extract registered depth and RGB images from point cloud and save them
    for topic, msg, t in bag_og.read_messages():
        if topic == "/throttled/camera/depth_registered/points":
            new_pc = PointCloud2()
            new_pc.header = msg.header
            new_pc.height = msg.height
            new_pc.width = msg.width
            new_pc.fields = msg.fields
            new_pc.is_bigendian = msg.is_bigendian
            new_pc.point_step = msg.point_step
            new_pc.row_step = msg.row_step
            new_pc.is_dense = msg.is_dense
            new_pc.data = msg.data

            pc = ros_numpy.numpify(new_pc)
            pc = ros_numpy.point_cloud2.split_rgb_field(pc).reshape(-1)
            depth = np.zeros((pc.shape[0], 1), dtype=np.uint16)
            depth[:, 0] = pc['z'] * 1000
            # Convert 0 to 50000
            # depth[depth == 0] = 50000

            rgb = np.zeros((pc.shape[0], 3))
            rgb[:, 0] = pc['r']
            rgb[:, 1] = pc['g']
            rgb[:, 2] = pc['b']

            # Convert rgb array to 640x480x3 image
            rgb_image = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
            rgb_image[:, :, 2] = np.uint8(rgb[:, 0].reshape(msg.height, msg.width))  # Red channel
            rgb_image[:, :, 1] = np.uint8(rgb[:, 1].reshape(msg.height, msg.width))  # Green channel
            rgb_image[:, :, 0] = np.uint8(rgb[:, 2].reshape(msg.height, msg.width))  # Blue channel

            depth_image = depth.reshape(msg.height, msg.width)

            # Convert RGB and depth images to ROS messages
            rgb_msg = cv_bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
            depth_msg = cv_bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")

            rgb_msg.header = msg.header
            depth_msg.header = msg.header

            # Write RGB and depth images to the bag file
            bag_out.write(rgb_topic, rgb_msg, t)
            bag_out.write(depth_topic, depth_msg, t)

        if topic in ["/throttled/camera/color/image_raw", "/throttled/camera/depth/image_rect_raw", "/throttled/camera/depth_registered/points"]:
            continue
        else:
            bag_out.write(topic, msg, t)

    bag_og.close()
    bag_out.close()

    print("Finished processing bag file: " + bag_file_name)


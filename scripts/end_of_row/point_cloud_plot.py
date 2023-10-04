#!/usr/bin/env python3


import rosbag
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

bag_file = '/media/jostan/MOAD/research_data/achyut_data/before_sept6/zed2_pcl.bag'

point_cloud_msgs = []

stopper = 0

bag = rosbag.Bag(bag_file)
for topic, msg, t in bag.read_messages(topics=['/zed2/zed_node/pointcloud/points']):
    if topic == '/zed2/zed_node/pointcloud/points' and stopper == 100:

        point_cloud_msg = msg
    stopper +=1
bag.close()
points = []

for point in pc2.read_points(point_cloud_msg, skip_nans=True):
    points.append(point)

# Convert the point cloud data to a numpy array
points = np.array(points)

# Plot the point cloud data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.05)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=-135, azim=90)
plt.show()


# # Define the size of the voxel
# voxel_size = 0.1
#
# # Define the planes perpendicular to the camera and their offset
# planes = [(1, 0, 0, 3), (0, 1, 0, -1)]
#
# # Calculate the maximum and minimum values for each dimension
# min_values = np.min(points, axis=0)
# max_values = np.max(points, axis=0)
#
# # Calculate the number of voxels along each dimension
# num_voxels_x = int(np.ceil((max_values[0] - min_values[0]) / voxel_size))
# num_voxels_y = int(np.ceil((max_values[1] - min_values[1]) / voxel_size))
# num_voxels_z = int(np.ceil((max_values[2] - min_values[2]) / voxel_size))
#
# # Create an array to store the voxels
# voxels = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z))
#
# # Divide the point cloud data into voxels
# for point in points:
#     x = int(np.floor((point[0] - min_values[0]) / voxel_size))
#     y = int(np.floor((point[1] - min_values[1]) / voxel_size))
#     z = int(np.floor((point[2] - min_values[2]) / voxel_size))
#     for plane in planes:
#         if abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) < voxel_size / 2:
#             voxels[x, y, z] += 1
#
# # Plot the voxels in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = np.nonzero(voxels)
# colors = voxels[x, y, z]
# norm = plt.Normalize(colors.min(), colors.max())
# ax.scatter(x * voxel_size + min_values[0], y * voxel_size + min_values[1], z * voxel_size + min_values[2], c=colors, cmap='cool', s=10, alpha=0.5, norm=norm)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
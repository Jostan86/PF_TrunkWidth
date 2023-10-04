#!/usr/bin/env python3


import rosbag
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bag_file = '../data/depth_data.bag'

point_cloud_msgs = []

stopper = 0

bag = rosbag.Bag(bag_file)
for topic, msg, t in bag.read_messages(topics=['/camera/depth/reconstructed/points']):
    if topic == '/camera/depth/reconstructed/points' and stopper == 100:

        point_cloud_msg = msg
    stopper +=1
bag.close()

# point_cloud_msg = point_cloud_msgs[0]

# Define the dimensions of the voxels
voxel_size = 0.1
x_min = 0.1
x_max = 0.5
y_min = -0.6
y_max = -0.2
z_min = 0.0
z_max = 2.5

# Initialize the voxel grid
x_bins = int((x_max - x_min) / voxel_size)
y_bins = int((y_max - y_min) / voxel_size)
z_bins = int((z_max - z_min) / voxel_size)
voxel_grid = [[[0 for k in range(z_bins)] for j in range(y_bins)] for i in range(x_bins)]

# Iterate through each point in the point cloud
for point in pc2.read_points(point_cloud_msg, skip_nans=True):
    x = point[0]
    y = point[1]
    z = point[2]

    # Check if the point is within the voxel grid
    if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
        x_idx = int((x - x_min) / voxel_size)
        y_idx = int((y - y_min) / voxel_size)
        z_idx = int((z - z_min) / voxel_size)
        voxel_grid[x_idx][y_idx][z_idx] += 1

# Print the number of points in each voxel along a plane perpendicular to the camera, but offset
for x_idx in range(x_bins):
    x = x_min + (x_idx + 0.5) * voxel_size
    y = 0.0  # The plane perpendicular to the camera is at y=0
    for z_idx in range(z_bins):
        z = z_min + (z_idx + 0.5) * voxel_size
        print("({:.1f}, {:.1f}, {:.1f}): {}".format(x, y, z, voxel_grid[x_idx][y_bins // 2][z_idx]))

# # Plot the voxels in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = np.nonzero(voxel_grid)
# colors = voxel_grid[x, y, z]
# norm = plt.Normalize(colors.min(), colors.max())
# ax.scatter(x * voxel_size + min_values[0], y * voxel_size + min_values[1], z * voxel_size + min_values[2], c=colors, cmap='cool', s=10, alpha=0.5, norm=norm)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
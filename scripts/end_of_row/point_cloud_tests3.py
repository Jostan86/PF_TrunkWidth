#!/usr/bin/env python3


import rosbag
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

bag_file = '../data/depth_data2.bag'

point_cloud_msgs = []

stopper = 0

bag = rosbag.Bag(bag_file)
for topic, msg, t in bag.read_messages(topics=['/camera/depth/reconstructed/points']):
    if topic == '/camera/depth/reconstructed/points' and stopper == 1:

        point_cloud_msg = msg
    stopper +=1
bag.close()
print(stopper)
# Convert the point cloud message to a numpy array
point_cloud = np.array(list(pc2.read_points(point_cloud_msg)))


# Define the yz region to plot
z_min = 0.4
z_max = 4
y_min = -0.5

y_max = 0.05 # set this to 0.05 to remove the ground plane, 0.2 to include the ground

# Define the x range to filter
# set to -0.5 and -0.1 if you want the other side
x_min = 0.1
x_max = 0.5

# Filter the point cloud based on the x range
filtered_point_cloud = point_cloud[(point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max)]

# Reflect the remaining points onto the yz plane
yz_points = filtered_point_cloud[:, [1, 2]]

# Filter the yz points based on the yz region
yz_points = yz_points[
    (yz_points[:, 0] >= y_min) & (yz_points[:, 0] <= y_max) & (yz_points[:, 1] >= z_min) & (yz_points[:, 1] <= z_max)]

# Plot the remaining points in the yz region
fig = plt.figure(figsize=(6, 4), dpi=80)
ax = fig.add_subplot(111)
ax.scatter(yz_points[:, 1], yz_points[:, 0]*-1, c='b', s=1)

ax.axis('equal')
ax.set_xlim(0, 2.5)
ax.set_ylim(-0.5, 0.5)


z_values = yz_points[:, 1]

# # Show the plot
# # Create a histogram of the z values
# bin_size = 0.05
# num_bins = int(np.ceil((z_max - z_min) / bin_size))
# counts, bin_edges = np.histogram(z_values, bins=num_bins, range=(z_min, z_max))
#
# # Find the index of the first bin with a count under 100
# index = np.where(counts < 100)[0][0]
# distance = z_min + (index + 1) * bin_size
#
#
# # # Plot the histogram
# plt.figure(figsize=(8, 4), dpi=80)
# plt.hist(z_values, bins=num_bins, range=(z_min, z_max))
plt.xlabel('Distance from Camera (m)')
plt.ylabel('Num Points')
plt.title('Right Side Points with Ground Removed')
plt.show()



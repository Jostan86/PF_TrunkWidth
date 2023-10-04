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
z_max = 5
y_min = -0.5

y_max = 0.2 # set this to 0.05 to remove the ground plane

# Define the x range to filter
# set to -0.5 and -0.1 if you want the other side
x_min = 0.1
x_max = 0.5


# Create a figure and a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Filter the point cloud based on the x range
filtered_point_cloud = point_cloud[(point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max)]

# Reflect the remaining points onto the yz plane
yz_points = filtered_point_cloud[:, [1, 2]]

# Filter the yz points based on the yz region
yz_points = yz_points[
    (yz_points[:, 0] >= y_min) & (yz_points[:, 0] <= y_max) & (yz_points[:, 1] >= z_min) & (yz_points[:, 1] <= z_max)]

# Plot the remaining points in the yz region
ax.scatter(yz_points[:, 1], yz_points[:, 0]*-1, c='b', s=1)

ax.axis('equal')


# Show the plot
plt.show()


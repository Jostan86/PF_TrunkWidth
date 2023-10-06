#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import String

class RowEndDetector():
    def __init__(self):
        rospy.init_node('row_end_detector')
        self.msg_pub = rospy.Publisher('/row_end_distance', String, queue_size=1)
        self.pcl_sub = rospy.Subscriber('/camera/depth/reconstructed/points', PointCloud2, self.pcl_callback)
        self.counter = 0
        self.num_to_skip = 5


    def pcl_callback(self, msg):
        if self.counter == self.num_to_skip:
            self.counter = 0
        else:
            self.counter += 1
            return

        point_cloud = np.array(list(pc2.read_points(msg)))

        # Define the yz region of interest
        z_min = 0.4
        z_max = 4
        y_min = -0.5

        y_max = 0.05 # set this to 0.05 to remove the ground plane, 0.2 to include the ground

        # Define the x range to filter
        # set to -0.5 and -0.1 if you want the other side
        x_min = 0.1
        x_max = 0.6

        # Filter the point cloud based on the x range
        filtered_point_cloud = point_cloud[(point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max)]

        # Reflect the remaining points onto the yz plane
        yz_points = filtered_point_cloud[:, [1, 2]]

        # Filter the yz points based on the yz region
        yz_points = yz_points[
            (yz_points[:, 0] >= y_min) & (yz_points[:, 0] <= y_max) & (yz_points[:, 1] >= z_min) & (yz_points[:, 1] <= z_max)]

        # Show the plot
        # Create a histogram of the z values
        bin_size = 0.05
        num_bins = int(np.ceil((z_max - z_min) / bin_size))
        counts, bin_edges = np.histogram(yz_points[:, 1], bins=num_bins, range=(z_min, z_max))

        # Find the index of the first bin with a count under 100
        index = np.where(counts < 100)[0][0]

        distance = z_min + (index+1) * bin_size

        if distance >= 0.5:
            message = "The row ends in approximately " + str(round(distance, 2)) + "m"
        else:
            message = "The row ends in less than 0.5m"
        str_msg = String()
        str_msg.data = message
        self.msg_pub.publish(str_msg)


end_detector = RowEndDetector()

rospy.spin()




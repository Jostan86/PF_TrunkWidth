#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

# Script to help find time sync issues between computers

def odometry_callback(msg):
    # Get the timestamp from the odometry message header
    odom_time = msg.header.stamp.to_sec()

    # Get the current ROS time
    current_time = rospy.get_rostime().to_sec()

    # Calculate the time difference
    time_difference = current_time - odom_time

    # Print the time difference
    rospy.loginfo("Time difference between Odometry timestamp and current ROS time: %s", time_difference)

def header_callback(msg):
    header_time = msg.stamp.to_sec()
    current_time = rospy.get_rostime().to_sec()
    time_difference = current_time - header_time

    rospy.loginfo("Time difference between Header timestamp and current ROS time: %s", time_difference)
def main():
    # Initialize the node
    rospy.init_node('time_difference_node')

    # Subscribe to the odometry topic
    # rospy.Subscriber('/odometry/filtered', Odometry, odometry_callback)
    rospy.Subscriber('/ros_time', Header, header_callback)


    # Keep the script running
    rospy.spin()

if __name__ == '__main__':
    main()

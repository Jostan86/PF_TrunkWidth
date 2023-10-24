#!/usr/bin/env python3

# script to publish time to /ros_time_framework topic
# this is used to synchronize the time between the robot and the computer

import rospy
from std_msgs.msg import Header

def main():
    rospy.init_node('time_pub')
    pub = rospy.Publisher('/ros_time_framework', Header, queue_size=10)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        msg = Header()
        msg.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    main()
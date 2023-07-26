#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
import tf
from tf.transformations import quaternion_multiply, quaternion_inverse, quaternion_slerp

class PosePublisher:
    def __init__(self):
        rospy.init_node('odom_ground_truth')
        self.link_states_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.odom_callback)
        self.odom_pub = rospy.Publisher('/pose_gt', PoseStamped, queue_size=1)
        self.pose_msg = None
        # self.listener = tf.TransformListener()

    def odom_callback(self, msg):


        for i, name in enumerate(msg.name):
            if name == "husky::base_link":
                idx = i
                self.pose_msg = msg.pose[idx]
                odom_msg = PoseStamped()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = "map"
                odom_msg.pose = self.pose_msg
                self.odom_pub.publish(odom_msg)



    # def map_transform(self):
    #     if self.pose_msg is None:
    #         return
    #     self.listener.waitForTransform('/base_link', '/odom', rospy.Time(), rospy.Duration(1, 0))
    #     (trans, rot) = self.listener.lookupTransform('/base_link', '/odom', rospy.Time(0))
    #
    #     rot_diff = quaternion_multiply([self.pose_msg.orientation.x, self.pose_msg.orientation.y, self.pose_msg.orientation.z, self.pose_msg.orientation.w], quaternion_inverse(rot))
    #
    #     rot_diff_norm = quaternion_slerp((1.0, 0.0, 0.0, 0.0), rot_diff, 1.0)
    #
    #     br = tf.TransformBroadcaster()
    #     br.sendTransform(
    #         (self.pose_msg.position.x - trans[0], self.pose_msg.position.y - trans[1], self.pose_msg.position.z -
    #          trans[2]),
    #         # (self.pose_msg.orientation.x, self.pose_msg.orientation.y, self.pose_msg.orientation.z, self.pose_msg.orientation.w),
    #         rot_diff_norm,
    #         rospy.Time.now(),
    #         "/odom",
    #         "/map"
    #
    #     )


pose_pub = PosePublisher()
# rate = rospy.Rate(10)
# try:
#     while not rospy.is_shutdown():
#         pose_pub.map_transform()
#         rate.sleep()
# except rospy.ROSInterruptException:
#     pass
#
rospy.spin()


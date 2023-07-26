#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from sim_pf_trunk_width.msg import TreeInfo, TreeSensor

class ImageFilter:

    def __init__(self):
        self.pub_images = False
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber('/realsense/color/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/realsense/depth/image_rect_raw', Image, self.depth_callback)
        if self.pub_images:
            self.filtered_pub = rospy.Publisher('/image/filtered', Image, queue_size=1)
            self.lines_pub = rospy.Publisher('/image/lines', Image, queue_size=1)

        self.tree_info_pub = rospy.Publisher('/tree_sensor/tree_info', TreeSensor, queue_size=1)

        self.depth_image_msg = None
        self.rgb_image_msg = None
        self.num_to_skip = 5
        self.count = 0

    def depth_callback(self, data):
        self.depth_image_msg = data
        self.pair_images()

    def rgb_callback(self, data):
        self.rgb_image_msg = data
        self.pair_images()


    def pair_images(self):
        if self.rgb_image_msg is None or self.depth_image_msg is None:
            return

        if self.rgb_image_msg.header.stamp != self.depth_image_msg.header.stamp:
            return

        # Only process every 5th image
        if self.count != self.num_to_skip:
            self.count += 1
            return
        else:
            self.count = 0

        # Convert depth image to a numpy array
        depth_image = self.bridge.imgmsg_to_cv2(self.depth_image_msg)

        # Make a copy of the image
        depth_mask = depth_image.copy()

        # Remove the nan values and replace them with 8
        depth_mask[np.isnan(depth_mask)] = 8

        # Scale the image to be 0 - 255
        max = np.amax(depth_mask)
        scale = 255.0 / max
        depth_mask *= scale
        depth_mask = np.round(depth_mask).astype(int)
        depth_array = cv2.convertScaleAbs(depth_mask, alpha=1)  # Scale depth values to 8-bit range

        # Make depth mask image, then reverse it
        _, depth_mask_reverse = cv2.threshold(depth_array, 80, 255, cv2.THRESH_BINARY)
        depth_mask = cv2.bitwise_not(depth_mask_reverse)


        # Convert RGB image to a numpy array
        rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_image_msg, desired_encoding='bgr8')

        # Convert to HSV space
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds of the color to filter
        lower = np.array([0, 102, 0])  # lower bound for H, S, V values
        upper = np.array([20, 255, 127])  # upper bound for H, S, V values

        # Create a binary mask of the pixels within the color range
        hsv_mask = cv2.inRange(hsv_image, lower, upper)

        # Combine the hsv mask and the depth mask
        combined_mask = cv2.bitwise_and(depth_mask, hsv_mask)

        # Do some morphology to get rid of the noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Apply the mask to the original image using the bitwise AND operator
        # image_masked = cv2.bitwise_and(rgb_image, rgb_image, mask=combined_mask)

        # Apply edge detection to the combined mask image
        edges = cv2.Canny(combined_mask, 50, 150, apertureSize=3)

        # Apply the Hough Line Transform to the edge image
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=150)

        # Create a white image with the same size as the original image
        white = np.ones_like(combined_mask) * 255

        # Get the pixel locations of the trees
        rho_list = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                rho_list.append(rho)

        rho_list.sort()

        # Get the pairs of tree edges, it seems to basically always return both edges, so if there are an odd
        # number of edges it just needs to determine whether the first or last edge is by itself
        tree_edges_locations = []
        if len(rho_list) % 2 == 0:
            # If even number of edges, just pair them off
            for i in range(int(len(rho_list)/2)):
                tree_edges_locations.append([rho_list[2*i], rho_list[2*i+1]])

        else:
            # If odd, calculate the sum of the widths while ignoring the first and the last, and choose the smaller
            # value
            ignore_first = []
            ignore_last = []
            diff_first = 0
            diff_last = 0
            for i in range(int(len(rho_list[1:])/2)):
                ignore_first.append([rho_list[2 * i + 1], rho_list[2 * i + 2]])
                diff_first += rho_list[2 * i + 2] - rho_list[2 * i + 1]
            for i in range(int(len(rho_list[:-1])/2)):
                ignore_last.append([rho_list[2 * i], rho_list[2 * i + 1]])
                diff_last += rho_list[2 * i + 1] - rho_list[2 * i]
            if diff_last < diff_first:
                tree_edges_locations = ignore_last[:]
            else:
                tree_edges_locations = ignore_first[:]

        # Loop over the tree edge locations and calculate the tree width, angle, and depth
        tree_info_list = []
        for tree_edges in tree_edges_locations:

            # Get tree edge pixel locations
            tree_edge_1 = tree_edges[0] - 1
            tree_edge_2 = tree_edges[1]

            # Calculate tree width in pixels
            tree_width_pixel = int(tree_edge_2 - tree_edge_1)

            # Find the center pixels of the tree
            tree_cen_pix_x = int(tree_edge_1 + tree_width_pixel/2)
            tree_cen_pix_y = int(self.rgb_image_msg.height / 2)

            # Find the center pixels of the image along the x axis
            image_cent_x = int(self.rgb_image_msg.width / 2)

            # Calculate how far off center the middle and each edge of the tree is, in pixels
            num_pixels_off_center = tree_cen_pix_x - image_cent_x
            edge1_off_center = (tree_edge_1 - image_cent_x)/ image_cent_x
            edge2_off_center = (tree_edge_2 - image_cent_x)/ image_cent_x

            # Calculate the angle to each edge and the center
            edge1_angle = np.arctan(edge1_off_center * np.tan(np.radians(43.5)))
            edge2_angle = np.arctan(edge2_off_center * np.tan(np.radians(43.5)))
            angle_center = np.arctan((num_pixels_off_center/ (image_cent_x)) * np.tan(np.radians(43.5)))
            # Calculate the total angle that the tree takes up
            angle = abs(edge1_angle - edge2_angle)

            # Calculate the tree depth
            tree_depth = depth_image[tree_cen_pix_y, tree_cen_pix_x]/np.cos(angle_center)

            # Calculate the tree width
            tree_width = np.tan(angle/2) * tree_depth*2
            # Recalculate the tree depth, now to center of tree
            tree_depth = depth_image[tree_cen_pix_y, tree_cen_pix_x] / np.cos(angle_center) + tree_width/2
            # Recalculate width based on updated depth
            tree_width = np.tan(angle/2) * tree_depth*2

            tree_info_msg = TreeInfo()
            tree_info_msg.width = tree_width
            tree_info_msg.distance = tree_depth
            tree_info_msg.angle = -angle_center

            tree_info_list.append(tree_info_msg)

        tree_sensor_msg = TreeSensor()
        # tree_sensor_msg.header.frame_id = ?
        tree_sensor_msg.header.stamp = rospy.Time.now()
        tree_sensor_msg.trees = tree_info_list

        self.tree_info_pub.publish(tree_sensor_msg)

        if self.pub_images:
            # Create a white image with the same size as the original image
            white = np.ones_like(combined_mask) * 255

            # white[edges != 0] = 0
            # Draw the detected lines on the original image
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(white, (x1, y1), (x2, y2), 0, 2)

            # Publish filtered image
            filtered_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            white_msg = self.bridge.cv2_to_imgmsg(white, encoding='mono8')

            self.filtered_pub.publish(filtered_msg)
            self.lines_pub.publish(white_msg)

        # Apply image processing here, e.g., use depth image to filter RGB image

        # Reset images
        self.rgb_image_msg = None
        self.depth_image_msg = None




if __name__ == '__main__':
    rospy.init_node('image_filter_node')
    filter = ImageFilter()
    rospy.spin()
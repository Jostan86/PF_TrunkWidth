import os
import sys
import cv2
import numpy as np

import numpy as np
from scipy.spatial import distance
from skimage import morphology
from ultralytics import YOLO
import cv2
from skimage.measure import label, regionprops
import time
import torch

class TrunkAnalyzer:
    def __init__(self):

        # Initialize predictor using the configuration object
        # weight_path = "/home/jostan/OneDrive/Docs/Grad_school/Research/yolo_model/best_x_500_v5.pt"
        weight_path = "/home/jostan/OneDrive/Docs/Grad_school/Research/yolo_model/best_x_500_v7.pt"
        self.yolo_model = YOLO(weight_path)

        self.results = None
        self.image = None
        self.depth_img = None

        self.confidences = None
        self.masks = None
        self.classes = None
        self.results_kept = None

        self.depth_calculated = False
        self.depth_median = None
        self.depth_percentile_upper = None
        self.depth_percentile_lower = None

        self.width_calculated = False
        self.tree_widths = None

        # self.straightness_metrics_calculated = False
        # self.angle_diff = None
        # self.mean_deviation_sum = None
        # self.mean_length = None
        # self.mean_deviation_left = None
        # self.mean_deviation_right = None
        # self.lines = None

        # self.color_calculated = False
        # self.hsv_value_channel = None

        self.num_instances = None
        self.width = None
        self.height = None
        self.num_pixels = None

        # self.classification_calculated = False
        # self.classification = None

        self.tree_locations = None

        self.save_count = 0

    def get_mask(self, image, remove_sprinklers=True):
        """
        Do the prediction and save the mask and score variables
        Args:
            image (): Image that was sent to the model.

        Returns:
            saves the mask and score arrays to the class variables, as well as the number of instances, and the indices
            of the instances that were kept
        """
        # Do the prediction and convert to cpu
        # time_start = time.time()
        results = self.yolo_model.predict(image, imgsz=(image.shape[0], image.shape[1]), iou=0.01, conf=0.5, verbose=False)
        # results = self.yolo_model.predict(image, iou=0.01, conf=0.5)

        # time_end = time.time()
        # print("Prediction time: ", time_end - time_start)

        self.results = results[0].cpu()

        # Save the number of instances
        self.num_instances = len(self.results.boxes) if len(self.results.boxes) > 0 else None

        # Save the mask and score arrays to the class variables
        self.confidences = self.results.boxes.conf.numpy() if self.num_instances is not None else None
        self.masks = self.results.masks.data.numpy().astype(np.uint8) if self.num_instances is not None else None
        self.classes = self.results.boxes.cls.numpy() if self.num_instances is not None else None

        # Initialize the array that will hold the indices of the instances that are being kept
        self.results_kept = np.arange(self.num_instances) if self.num_instances is not None else None

        if remove_sprinklers and self.num_instances is not None:
            # Remove any masks of class 1
            keep_indices = self.classes != 1
            self.update_arrays(keep_indices)


    def update_arrays(self, keep_indices):
        """Update all the arrays based on the keep_indices array from a filtering operation.

        Args:
            keep_indices (): Indices of the instances that are being kept.

        Returns:
            Filters all the active arrays based on the keep_indices array.
        """

        self.confidences = self.confidences[keep_indices]
        self.masks = self.masks[keep_indices]
        self.results_kept = self.results_kept[keep_indices]
        self.classes = self.classes[keep_indices]
        self.num_instances = len(self.masks)
        self.results = self.results[keep_indices]

        if self.depth_calculated:
            self.depth_median = self.depth_median[keep_indices]
            self.depth_percentile_upper = self.depth_percentile_upper[keep_indices]
            self.depth_percentile_lower = self.depth_percentile_lower[keep_indices]

        if self.width_calculated:
            self.tree_widths = self.tree_widths[keep_indices]
            self.tree_locations = self.tree_locations[keep_indices]

    def get_width_pix(self, mask):

        # Get the medial axis and distance transform of the mask.
        # The distance transform assigns to each pixel the distance to the closest background pixel.
        # The medial axis is the set of pixels that are equidistant to two or more background pixels. So for a
        # rectangle it would be the center line of the rectangle with ys at the top and bottom towards the corners.
        medial_axis, return_distance = morphology.medial_axis(mask, return_distance=True)

        # Get the number of medial axes in each row
        axes_per_row = medial_axis.sum(axis=1)

        # Get the indices of the rows that have a medial axis value of one
        one_indices = np.nonzero(axes_per_row == 1)[0]

        # Find the differences between consecutive non-zero indices
        consecutive_diffs = np.diff(one_indices)

        # Identify the start and end indices of the continuous segments
        change_indices = np.nonzero(consecutive_diffs > 1)[0]
        start_indices = np.append(np.array([0]), change_indices + 1)
        end_indices = change_indices
        end_indices = np.append(end_indices, len(one_indices) - 1)

        # Find the longest segment
        segment_lengths = end_indices - start_indices
        longest_segment_index = np.argmax(segment_lengths)

        # Get the start and end index of the longest segment
        mstart = one_indices[start_indices[longest_segment_index]]
        mend = one_indices[end_indices[longest_segment_index]]
        mlen = segment_lengths[longest_segment_index]

        # Make array of zeros the same size as the number of rows in the mask, with ones in the rows that have a medial axis
        medial_axis_mask = np.zeros_like(axes_per_row)
        medial_axis_mask[mstart:mend] = 1

        # Get the return distance of the pixels along the medial axis
        return_distance_masked = return_distance * medial_axis_mask[:, np.newaxis].astype(bool)
        return_distance_axis = np.max(return_distance_masked, axis=1)
        return_distance_axis = return_distance_axis[mstart:mend]

        cut_off_dist = int(mlen * 0.2)
        return_distance_axis_trimmed = np.cumsum(return_distance_axis)[:-cut_off_dist]
        return_distance_axis_trimmed = return_distance_axis_trimmed[cut_off_dist:]

        # Calculate the difference between elements separated by a window of 20 in return_distance_axis_trimmed (effectively a discrete derivative)
        return_distance_axis_derv = return_distance_axis_trimmed[20:] - return_distance_axis_trimmed[:-20]

        # Determine how many of the smallest elements in return_distance4 to consider (40% of the length of the array)
        k = int(return_distance_axis_derv.shape[0] * 0.4)

        # Find the indices of the k smallest elements in return_distance_axis_derv
        idx1 = np.argpartition(return_distance_axis_derv, k)[:k]

        # Calculate the real indices in the original return_distance_axis array
        # by accounting for the shift introduced when computing return_distance4 (10 positions), and the offset
        # introduced when removing the top and bottom 20% of values
        real_idx = idx1 + 10 + cut_off_dist

        # Retrieve the distances at the calculated indices from return_distance1 and double them to estimate the trunk diameter in pixels at those points
        diameter_pixels = return_distance_axis[real_idx] * 2

        return diameter_pixels


    def calculate_depth(self, top_ignore=0.4, bottom_ignore=0.20, min_num_points=300,
                        depth_filter_percentile_upper=65, depth_filter_percentile_lower=35):
        """
        Calculates the best estimate of the distance between the tree and camera.

        Args:
            top_ignore (): Proportion of the top of the image to ignore mask points in.
            bottom_ignore (): Proportion of the bottom of the image to ignore mask points in.
            min_num_points (): Minimum number of valid pointcloud points needed to keep the mask, if less than this,
            disregard the mask.
            depth_filter_percentile (): Percentile of the depth values to use for the percentile depth estimate. So at
            65, the percentile depth will be farther than 65% of the points in the mask.

        Returns:
            Calculates the median depth and percentile depth for each mask. Also filters out masks that have less than
            min_num_points valid points in the region defined by top_ignore and bottom_ignore.

        """

        # Initialize arrays to store the depth values and the tree locations
        self.depth_median = np.zeros(self.num_instances)
        self.depth_percentile_upper = np.zeros(self.num_instances)
        self.depth_percentile_lower = np.zeros(self.num_instances)

        # Make boolean array of indices to keep
        keep = np.ones(self.num_instances, dtype=bool)

        # Replace all nan values with 0
        # self.pointcloud = np.where(np.isnan(self.pointcloud), 0, self.pointcloud)
        #
        # # Reshape the point cloud array to match the mask dimensions
        # reshaped_cloud = self.pointcloud.reshape(-1, 3)

        # Calculate the top and bottom ignore values in pixels
        top_ignore = int(top_ignore * self.height)
        bottom_ignore = self.height - int(bottom_ignore * self.height)

        # Loop through each mask
        for i, mask in enumerate(self.masks):

            # Make copy of mask array
            mask_copy = mask.copy()

            # Zero out the top and bottom ignore regions
            mask_copy[:top_ignore, :] = 0
            mask_copy[bottom_ignore:, :] = 0

            # If there are no points in the mask, remove the segment
            if np.sum(mask_copy) == 0:
                keep[i] = False
                continue

            # Convert mask copy to a boolean array
            mask_copy = mask_copy.astype(bool)

            # Make a 1D array of the masked portions of the depth image, which is currently a 2d array
            masked_depth = self.depth_img[mask_copy]

            # Remove zero values from the masked depth array
            masked_depth = masked_depth[masked_depth != 0]

            # If there are less than the min number of points, remove the mask
            if masked_depth.shape[0] < min_num_points:
                keep[i] = False
                continue

            # Calculate median depth
            self.depth_median[i] = np.median(masked_depth) / 1000
            # Calculate the percentile depth
            self.depth_percentile_upper[i] = np.percentile(masked_depth, depth_filter_percentile_upper) / 1000
            self.depth_percentile_lower[i] = np.percentile(masked_depth, depth_filter_percentile_lower) / 1000

            # print("Upper, Median, Lower: ", self.depth_percentile_upper[i], self.depth_median[i], self.depth_percentile_lower[i])
            #

        # Update the arrays
        self.depth_calculated = True
        self.update_arrays(keep)

    def calculate_width(self):
        """
        Calculates the best estimate of the width of the tree in meters.

        Args:
            horz_fov (): Horizontal field of view of the camera in degrees.

        Returns:
            Calculates and stores the width of the tree in meters for each mask.
        """

        self.tree_widths = np.zeros(self.num_instances)
        self.tree_locations = np.zeros((self.num_instances, 2))

        # Loop through each mask
        for i, (mask, depth) in enumerate(zip(self.masks, self.depth_median)):

            # Get the diameter of the tree in pixels
            diameter_pixels = self.get_width_pix(mask)

            # get image width in pixels
            image_width_pixels = mask.shape[1]
            if image_width_pixels == 640:
                horz_fov = 55.0
            elif image_width_pixels == 848 or image_width_pixels == 1280:
                horz_fov = 69.4
            else:
                print("Image width not supported, using default 69.4 degrees")
                horz_fov = 69.4

            # Calculate the width of the image in meters at the depth of the tree
            image_width_m = depth * np.tan(np.deg2rad(horz_fov / 2)) * 2

            # Calculate the distance per pixel
            distperpix = image_width_m / self.width

            # Calculate the diameter of the tree in meters
            diameter_m = diameter_pixels * distperpix

            # If there are no valid widths, set the width to 0, otherwise set it to the max width
            if len(diameter_m) == 0:
                self.tree_widths[i] = 0
            else:
                self.tree_widths[i] = np.max(diameter_m)

            # Calculate the x location of the tree in the image by taking the median of the mask points in x
            x_median_pixel = np.median(np.where(mask)[1])
            self.tree_locations[i, 1] = self.depth_median[i]
            self.tree_locations[i, 0] = (x_median_pixel - (self.width / 2)) * distperpix

        self.width_calculated = True


    def mask_filter_nms(self, overlap_threshold=0.5):
        """
        Apply non-maximum suppression (NMS) to a set of masks and scores.

        Args:
            overlap_threshold (): Overlap threshold for NMS. If the overlap between two masks is greater than this
            value, the mask with the lower score will be suppressed.

        Returns:
            Updates the class arrays to only include the masks that were not suppressed.
        """

        mask_nms = self.masks.copy()
        score_nms = self.confidences.copy()

        # Sort masks by score
        indices = np.argsort(-score_nms)
        mask_nms = mask_nms[indices]

        # Array to keep track of whether an instance is suppressed or not
        suppressed = np.zeros((len(mask_nms)), dtype=bool)

        # For each mask, compute overlap with other masks and suppress overlapping masks if their score is lower
        for i in range(len(mask_nms) - 1):
            # If already suppressed, skip
            if suppressed[i]:
                continue
            # Compute overlap with other masks
            overlap = np.sum(mask_nms[i] * mask_nms[i + 1:], axis=(1, 2)) / np.sum(mask_nms[i] + mask_nms[i + 1:],
                                                                                   axis=(1, 2))
            # Suppress masks that are either already suppressed or have an overlap greater than the threshold
            suppressed[i + 1:] = np.logical_or(suppressed[i + 1:], overlap > overlap_threshold)

        # Get the indices of the masks that were not suppressed
        indices_revert = np.argsort(indices)
        suppressed = suppressed[indices_revert]
        not_suppressed = np.logical_not(suppressed)

        # Update the arrays
        self.update_arrays(not_suppressed)

    def mask_filter_depth(self, depth_threshold=1.5):
        """Sort out any outputs that are beyond a given depth threshold. Note that during depth calcuation any
        segments entirely in the top or bottom portions of the image are removed, and any segments with too few
        points in the point cloud are also removed.

        Args:
            depth_threshold (): Depth threshold in meters. Any masks with a percentile depth greater than this value
            will be removed.

        Returns:
            Updates the class arrays to only include the masks that are within the depth threshold.
        """
        keep = self.depth_percentile_upper < depth_threshold
        self.update_arrays(keep)

    # def mask_filter_size(self, large_threshold=0.05, small_threshold=0.01, score_threshold=0.1):
    #     """Sort out any outputs with masks smaller or larger than the thresholds, based on number of pixels. Also,
    #     this filter ignores any masks with a score higher than the score threshold"""
    #
    #     keep = np.zeros(self.num_instances, dtype=bool)
    #
    #     large_threshold = large_threshold * self.num_pixels
    #     small_threshold = small_threshold * self.num_pixels
    #
    #     for i, (score, mask) in enumerate(zip(self.confidences, self.masks)):
    #         if score > score_threshold:
    #             keep[i] = True
    #             continue
    #         area = np.sum(mask)
    #         if area > large_threshold or area < small_threshold:
    #             continue
    #         else:
    #             keep[i] = True
    #
    #     self.update_arrays(keep)

    def mask_filter_edge(self, edge_threshold=0.05, size_threshold=0.1):
        """Sort out any outputs with masks that are too close to the edge of the image. Edge threshold is how close
        the mask can be to the edge, as a proportion of the image width. Size threshold is the proportion of the mask
        that must be beyond the edge threshold for the mask to be removed. """

        keep = np.zeros(self.num_instances, dtype=bool)

        edge_threshold = int(edge_threshold * self.width)

        masks_copy = self.masks.copy()

        for i, mask in enumerate(masks_copy):
            left_edge_pixels = mask[:, :edge_threshold].sum()
            right_edge_pixels = mask[:, -edge_threshold:].sum()
            total_mask_pixels = np.sum(mask)
            if left_edge_pixels / total_mask_pixels > size_threshold or right_edge_pixels / total_mask_pixels > size_threshold:
                continue
            else:
                keep[i] = True

        self.update_arrays(keep)

    def mask_filter_position(self, bottom_position_threshold=0.33, score_threshold=0.9, top_position_threshold=0.3):
        """Filter out any masks whose lowest point is above the bottom position threshold. Position threshold is the
        proportion of the image height from the bottom. Also filter out any masks that are entirely below the top
        position threshold."""

        keep = np.zeros(self.num_instances, dtype=bool)

        bottom_position_threshold = int(bottom_position_threshold * self.height)
        top_position_threshold = int(top_position_threshold * self.height)

        masks_copy = self.masks.copy()

        for i, mask in enumerate(masks_copy):
            # if self.confidences[i] > score_threshold:
            #     keep[i] = True
            #     continue

            bottom_pixels = mask[-bottom_position_threshold:].sum()
            if bottom_pixels > 0:
                keep[i] = True

            top_pixels = mask[:top_position_threshold].sum()
            if np.isclose(top_pixels, 0):
                keep[i] = False

        self.update_arrays(keep)

    def mask_filter_multi_segs(self):
        """Determine if there are multiple segments in the mask and if so keep only the largest one."""

        # Initialize an empty array to store the largest segments
        largest_segments = np.zeros_like(self.masks)

        # Loop through each mask
        for i, mask in enumerate(self.masks):
            # Label connected regions in the mask
            labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)

            # If there's only one connected region, no need to process further
            if num_labels == 1:
                largest_segments[i] = mask
            else:
                # Find properties of each connected region
                props = regionprops(labeled_mask)

                # Sort the regions by their area in descending order
                props.sort(key=lambda x: x.area, reverse=True)

                # Keep only the largest connected segment
                largest_segment_mask = labeled_mask == props[0].label

                # Store the largest segment in the result array
                largest_segments[i] = largest_segment_mask.astype(np.uint8)

        self.masks = largest_segments
        mask_ten = torch.from_numpy(self.masks)
        self.results.update(masks=mask_ten)

    # def mask_find_stacked(self, masks, scores):
    #     """Sort out any masks that are stacked vertically, indicating that they are the same object. Keep the trunk
    #     with the highest score."""
    #
    #     # masks = self.masks.copy()
    #     # scores = self.confidences.copy()
    #
    #     # Sort masks based on scores in descending order
    #     sorted_indices = np.argsort(scores)[::-1]
    #     sorted_masks = masks[sorted_indices]
    #
    #     filtered_masks = []
    #
    #     # Array to keep track of whether an instance is suppressed or not
    #     suppressed = np.zeros((len(masks)), dtype=bool)
    #
    #     for i in range(len(sorted_masks)):
    #
    #         # Sum the columns into a single row
    #         mask_sum = np.sum(sorted_masks[i], axis=0)
    #
    #         for j in range(len(filtered_masks)):
    #             filtered_sum = np.sum(filtered_masks[j], axis=0)
    #
    #             if np.any(np.logical_and(mask_sum, filtered_sum)):
    #                 suppressed[i] = True
    #                 break
    #
    #         if not suppressed[i]:
    #             filtered_masks.append(sorted_masks[i])
    #
    #     # Restore original ordering of suppressed indices
    #     indices_revert = np.argsort(sorted_indices)
    #     suppressed = suppressed[indices_revert]
    #
    #     return suppressed



    def new_image_reset(self, image, depth_image):
        """ Resets the class variables for a new image, and loads the new image and pointcloud.

        Args:
            image (): Image to be processed
            depth_image (): aligned depth image that corresponds to the image
        """
        self.image = image
        self.depth_img = depth_image

        self.confidences = None
        self.masks = None
        self.results_kept = None

        self.depth_calculated = False
        self.depth_median = None
        self.depth_percentile_upper = None
        self.tree_locations = None

        self.width_calculated = False
        self.tree_widths = None

        self.height = image.shape[0]
        self.width = image.shape[1]
        self.num_pixels = self.height * self.width

        self.get_mask(image)

    # def process_image(self):
    #
    #     if self.masks is None:
    #         return
    #
    #     # Send the masks through all the filters, skipping to the end if the number of instances is 0
    #     self.mask_filter_score(score_threshold=0.005)
    #     if self.num_instances > 0:
    #         self.mask_filter_nms(overlap_threshold=0.01)
    #     if self.num_instances > 0:
    #         self.calculate_depth(top_ignore=0.05, bottom_ignore=0.05, min_num_points=500,
    #                              depth_filter_percentile_upper=65, depth_filter_percentile_lower=35)
    #     if self.num_instances > 0:
    #         self.mask_filter_depth(depth_threshold=2.5)
    #     if self.num_instances > 0:
    #         self.mask_filter_edge(edge_threshold=0.03)
    #     if self.num_instances > 1:
    #         self.mask_filter_stacked()
    #     if self.num_instances > 0:
    #         self.mask_filter_position(bottom_position_threshold=0.3, score_threshold=0.9)
    #     if self.num_instances > 0:
    #         self.calculate_straightness_stats(top_remove=0.1, bottom_remove=0.05, num_points=1000)  # was 1500
    #         self.mask_filter_straightness(straightness_threshold=10, score_threshold=0.05)
    #
    #         self.find_avg_color(self.image, top_cutoff=0.25, bottom_cutoff=0.1)
    #
    #         self.calculate_width()
    #
    #         self.classify_posts_sprinklers()

    def process_image_husky(self):

        if self.masks is None:
            return

        # Send the masks through all the filters, skipping to the end if the number of instances is 0
        if self.num_instances > 0:
             self.mask_filter_multi_segs()
        if self.num_instances > 1:
            self.mask_filter_nms(overlap_threshold=0.5)
        if self.num_instances > 0:
            self.calculate_depth(top_ignore=0.50, bottom_ignore=0.20, min_num_points=500,
                                 depth_filter_percentile_upper=65, depth_filter_percentile_lower=35)
        if self.num_instances > 0:
            self.mask_filter_depth(depth_threshold=2.0)

        if self.num_instances > 0:
            self.mask_filter_edge(edge_threshold=0.05)
        # if self.num_instances > 1:
        #     self.mask_filter_stacked()
        if self.num_instances > 0:
            self.mask_filter_position(bottom_position_threshold=0.5, top_position_threshold=0.65, score_threshold=0.9)

            # self.calculate_width()
            self.calculate_width()



    # def show_filtered_masks(self, image, depth_image):
    #     """ Show all the masks and the results after each filter."""
    #     self.new_image_reset(image, depth_image)
    #     if self.masks is None:
    #         return
    #     self.show_current_output("og")
    #     self.mask_filter_score(score_threshold=0.005)
    #     self.show_current_output("score")
    #
    #     if self.num_instances > 0:
    #         self.mask_filter_nms(overlap_threshold=0.01)
    #         self.show_current_output("Score, NMS")
    #
    #     if self.num_instances > 0:
    #         self.calculate_depth(top_ignore=0.05, bottom_ignore=0.05, min_num_points=500,
    #                              depth_filter_percentile_upper=65, depth_filter_percentile_lower=35)  # was 300
    #         # self.calculate_depth(top_ignore=0.25, bottom_ignore=0.2, min_num_points=200, depth_filter_percentile=75)
    #         self.show_current_output("Score, NMS, Depth")
    #
    #     if self.num_instances > 0:
    #         self.mask_filter_depth(depth_threshold=2.5)
    #         self.show_current_output("Score, NMS, Depth, Depth Filter")
    #
    #     if self.num_instances > 0:
    #         self.mask_filter_size(large_threshold=0.05, small_threshold=0.01)
    #         self.show_current_output("Score, NMS, Depth, Depth Filter, Size")
    #
    #     if self.num_instances > 0:
    #         self.mask_filter_edge(edge_threshold=0.03)
    #         self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge")
    #
    #     if self.num_instances > 1:
    #         self.mask_filter_stacked()
    #         self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge, Stacked")
    #
    #     if self.num_instances > 0:
    #         self.mask_filter_position(bottom_position_threshold=0.3, score_threshold=0.9)
    #         self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge, Stacked, Position")
    #
    #     if self.num_instances > 0:
    #         self.calculate_straightness_stats(top_remove=0.1, bottom_remove=0.05, num_points=1000)  # was 1500
    #         self.mask_filter_straightness(straightness_threshold=10, score_threshold=0.05)
    #         self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge, Stacked, Position, Straightness")
    #
    #         self.find_avg_color(self.image, top_cutoff=0.25, bottom_cutoff=0.1)
    #
    #         self.calculate_width()
    #
    #         self.classify_posts_sprinklers()
    #
    #     # Print some stats about each mask, if there are any, and draw lines of best fit on the image.
    #     if self.num_instances > 0:
    #
    #         for i, mask in enumerate(self.masks):
    #
    #             # Figure out the x location of the mask
    #             mean_x = np.mean(np.where(mask)[1])
    #
    #             if self.classification[i] == 0:
    #                 print("Tree at x = {}".format(mean_x))
    #             elif self.classification[i] == 1:
    #                 print("Post at x = {}".format(mean_x))
    #             elif self.classification[i] == 2:
    #                 print("Sprinkler at x = {}".format(mean_x))
    #
    #             print("Width = {}".format(self.tree_widths[i]))
    #             print("x: {}, z: {}".format(round(self.tree_locations[i][0], 3), self.tree_locations[i][1]))
    #             print("--------------------")
    #
    #         # get mask from output 6, draw lines on it, then draw on image
    #         for line in self.lines:
    #             # Define the two endpoints of the line
    #             point1 = (line[0][0], line[0][1])
    #             point2 = (line[-1][0], line[-1][1])
    #
    #             # Reshape the points into the required format
    #             line_points = np.array([point1, point2], dtype=np.int32).reshape((-1, 1, 2))
    #
    #             # Draw the line on the canvas
    #             image = cv2.polylines(image, [line_points], isClosed=False, color=(0, 255, 0), thickness=2)
    #
    #     cv2.imshow('lines', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def show_current_output(self, label, save_path=None, return_img=False):


        plot = self.results.plot()

        if return_img:
            return plot
        else:
            cv2.imshow(label, plot)

        if save_path is not None:
            cv2.imwrite(save_path, plot)
            # Save the mask as a binary image
            # masks = self.masks
            # for l, mask in enumerate(masks):
            #     mask = mask.astype(np.uint8)
            #     mask = mask * 255
            #     save_path1 = save_path + 'mask{}.png'.format(l)
            #     cv2.imwrite(save_path1, mask)

    # def eval_helper(self, image, depth_image):
    #     self.new_image_reset(image, depth_image)
    #     # remove 8 pixels on each side
    #
    #     self.process_image()
    #
    #     # Get index of lowest absolute medial x value
    #     min_x_index = np.argmin(np.abs(self.tree_locations[:, 0]))
    #     keep = np.zeros(self.num_instances, dtype=bool)
    #     keep[min_x_index] = True
    #     self.update_arrays(keep)
    #
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     v0 = Visualizer(image_rgb, scale=1.0, )
    #     outputs0 = self.results[self.results_kept]
    #     out0 = v0.draw_instance_predictions(outputs0)
    #     image_masked = out0.get_image()[:, :, ::-1]
    #
    #     return self.tree_widths[0], image_masked, self.masks[0]

    def eval(self, image, depth_image):
        image = image[:, 8:-8]
        depth_image = depth_image[:, 8:-8]
        self.new_image_reset(image, depth_image)
        self.process_image_husky()

        print("Time taken: ", time.time() - time_start)
        self.show_current_output("segmented image")

        print("Tree Widths: ", self.tree_widths)
        print("Locations: ", self.tree_locations)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



        # return self.tree_widths[0], image_masked, self.masks[0]

    def pf_helper(self, image, depth_image, show_seg=False, save_path=None):
        self.new_image_reset(image, depth_image)
        self.process_image_husky()

        if self.num_instances == 0 or self.num_instances is None:
            # if show_seg:
            #     # Show the image
            #     cv2.imshow('Segmented image', image)
            #     cv2.waitKey(1)
            return None, None, None, image, None

        img_x_position = np.zeros(self.num_instances, dtype=np.int32)
        #
        for i, mask in enumerate(self.masks):
            # Figure out the x location of the mask
            img_x_position[i] = int(np.mean(np.where(mask)[1]))

        if show_seg:
            img = self.show_current_output("Segmented image", return_img=True, save_path=save_path)
            return self.tree_locations, self.tree_widths, self.classes, img, img_x_position
            # cv2.waitKey(1)
        else:
            return self.tree_locations, self.tree_widths, self.classes, img_x_position


if __name__ == "__main__":
    # estimator.plot_results(num)
    segmenation_model = TrunkAnalyzer()
    # image_rgb = cv2.imread('/media/jostan/portabits/bags_extracted/bag_3/images_depth/1603920864997508023_2.png')
    # image_depth = cv2.imread('/media/jostan/portabits/bags_extracted/bag_3/images_depth/1603920864997508023_2_d.png')
    image_rgb = cv2.imread('/media/jostan/MOAD/research_data/2020_orchard_data/November/Mapping_Data/afternoon2'
                           '/bags_extracted/bag_3/images_depth/1603920864997508023_2.png')
    depth_image_path = '/media/jostan/MOAD/research_data/2020_orchard_data/November/Mapping_Data/afternoon2/bags_extracted/bag_3/images_depth/1603920864997508023_2_d.png'
    image_depth = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    time_start = time.time()
    segmenation_model.eval(image_rgb, image_depth)


    # post = [3, 4, 20, 31, 62]
    #
    # post_98 = [21, 53, 62, 63, 73]
    # sprinkler = [33, 44, 59, 68, 74]
    #
    # # There is a wierd mask at img 279 of bag file 11
    # # Also one at 528 of bag file 0 if bottom_position_threshold is 0.33
    #
    # for i in range(400, 888):
    # # for i in sprinkler:
    #     img_directory = os.environ.get('DATASET_PATH') + 'bag_0/images_depth'
    #     file_list_rgb = glob.glob(os.path.join(img_directory, '*_{}.png'.format(i)))
    #     file_list_depth = glob.glob(os.environ.get('DATASET_PATH') + 'bag_0/images_depth/*_{}_d.png'.format(i))
    #
    #     if len(file_list_rgb) == 1 and len(file_list_depth) == 1:
    #         file_path = file_list_rgb[0]
    #         depth_path = file_list_depth[0]
    #         img = cv2.imread(file_path)
    #         depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    #     elif len(file_list_rgb) == 0:
    #         print("Error: No file found")
    #         break
    #     else:
    #         print("Error: More than one file found")
    #         break
    #     print(i)
    #     segmenation_model.save_count = i
    #     segmenation_model.show_filtered_masks(img, depth_img)




#!/usr/bin/env python3
import sys
from sensor_msgs.msg import Image
from pf_trunk_width.msg import TreeInfo, TreeInfoMulti, TreePosition
from live_width_estimation import TrunkAnalyzer, TrunkSegmenter
import numpy as np
import multiprocessing as mp
import os
import signal
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2

class TreeInfoPublisher:
    def __init__(self):

        self.bridge = CvBridge()

        rospy.init_node('tree_data_node', anonymous=True)

        # Get parameters
        # Max size of the queue for the image and results queues
        if not rospy.has_param('~queue_max_size'):
            rospy.logwarn("Parameter queue_max_size not found. Setting to 100")
            queue_max_size = 100
        else:
            queue_max_size = rospy.get_param('~queue_max_size')
            rospy.loginfo("Parameter queue_max_size found. Setting to {}".format(queue_max_size))

        # Number of frames to skip between each frame that is processed, to reduce the load on the system
        if not rospy.has_param('~skip_frames'):
            rospy.logwarn("Parameter skip_frames not found. Setting to 0")
            self.skip_frames = 0
        else:
            self.skip_frames = rospy.get_param('~skip_frames')
            rospy.loginfo("Parameter skip_frames found. Setting to {}".format(self.skip_frames))

        # Offset of the camera from the center of the robot
        if not rospy.has_param('~x_offset'):
            rospy.logwarn("Parameter x_offset not found. Setting to 0.8")
            self.x_offset = 0.8
        else:
            self.x_offset = rospy.get_param('~x_offset')
            rospy.loginfo("Parameter x_offset found. Setting to {}".format(self.x_offset))

        if not rospy.has_param('~y_offset'):
            rospy.logwarn("Parameter y_offset not found. Setting to -0.55")
            self.y_offset = -0.55
        else:
            self.y_offset = rospy.get_param('~y_offset')
            rospy.loginfo("Parameter y_offset found. Setting to {}".format(self.y_offset))

        self.rgb_msgs = []
        self.depth_msgs = []
        self.rgb_times = []
        self.rgb_time_stamps = []
        self.depth_times = []
        self.num_frames_skipped = 0

        # Setup queues and processes
        self.image_queue = mp.Queue(maxsize=queue_max_size)
        self.results_queue = mp.Queue(maxsize=queue_max_size)
        self.analyzed_results_queue = mp.Queue(maxsize=queue_max_size)
        self.msg_to_pub_queue = mp.Queue(maxsize=queue_max_size)
        self.error_queue = mp.Queue()

        self.inference_process = mp.Process(target=run_inference, args=(self.image_queue, self.results_queue,
                                                                        self.analyzed_results_queue,
                                                                        self.msg_to_pub_queue, self.error_queue))
        self.analyzer_process = mp.Process(target=process_results, args=(self.results_queue,
                                                                         self.analyzed_results_queue,
                                                                         self.error_queue, self.x_offset,
                                                                         self.y_offset))

        self.inference_process.start()
        self.analyzer_process.start()

        # Wait for the processes to initialize, mostly gives the inference process time to load the model
        rospy.sleep(3)

        self.start_times = []
        self.time_stamps = []

        # Setup publishers and subscribers

        self.data_pub = rospy.Publisher('/orchard_pf/trunk_data', TreeInfoMulti, queue_size=10)
        self.img_pub = rospy.Publisher('/orchard_pf/seg_image', Image, queue_size=10)

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

        rospy.on_shutdown(self.shutdown_hook)

    def image_callback(self, rgb_msg):
        # Save the time stamp and rgb image
        self.rgb_times.append(rgb_msg.header.stamp.to_sec())
        self.rgb_time_stamps.append(rgb_msg.header.stamp)
        self.rgb_msgs.append(rgb_msg)
        self.pair_imgs()

    def depth_callback(self, depth_msg):
        # Save the time stamp and depth image
        self.depth_times.append(depth_msg.header.stamp.to_sec())
        self.depth_msgs.append(depth_msg)
        self.pair_imgs()

    def pair_imgs(self):
        # Pair the images if the timesstamps match

        # Check if there is at least one rgb and depth image
        if len(self.rgb_times) == 0 or len(self.depth_times) == 0:
            return

        # Num to check sets how many images of each kind to check for a match. So basically if it's 2 it takes the
        # first 2 rgb images and the first 2 depth images and checks if any of them match. If they do it pairs them
        # and removes them from the list.
        num_to_check = 2

        idx = [i for i in range(min(num_to_check, len(self.rgb_times), len(self.depth_times)))]
        idx = np.meshgrid(idx, idx)

        for idx_d, idx_rgb in zip(idx[0].flatten(), idx[1].flatten()):
            depth_time = self.depth_times[idx_d]
            rgb_time = self.rgb_times[idx_rgb]

            if np.isclose(rgb_time, depth_time):

                # Convert the img messages to cv2 images
                try:
                    depth_image = self.bridge.imgmsg_to_cv2(self.depth_msgs[idx_d], "passthrough")
                    rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_msgs[idx_rgb], "bgr8")
                except CvBridgeError as e:
                    print(e)

                # Save the time stamp that the images were paired. This is used as the time stamp on the published
                # data. It could be better to use the image time stamp for that, but I found that those aren't always
                # reliable, as they seem to use the time from the computer that the image was taken on, which doesn't
                # nessesarily match the time elsewhere in the system.
                current_time = rospy.get_rostime()
                self.time_stamps.append(current_time)

                # Remove the images from the lists
                self.rgb_times = self.rgb_times[idx_rgb+1:]
                self.rgb_msgs = self.rgb_msgs[idx_rgb+1:]
                self.rgb_time_stamps = self.rgb_time_stamps[idx_rgb+1:]
                self.depth_times = self.depth_times[idx_d+1:]
                self.depth_msgs = self.depth_msgs[idx_d+1:]

                # Put the images in the queue to be processed, skipping frames if necessary
                if self.num_frames_skipped == self.skip_frames:
                    self.image_queue.put((rgb_image, depth_image, False))
                    self.num_frames_skipped = 0
                else:
                    self.num_frames_skipped += 1

    def publish_data(self):
        # Publish the data from the queue

        # Check if there is data in the queue
        if not self.msg_to_pub_queue.empty():
            # Publish all the data in the queue
            for i in range(self.msg_to_pub_queue.qsize()):
                img, tree_data = self.msg_to_pub_queue.get()

                # Get the oldest time stamp
                time_stamp = self.time_stamps.pop(0)

                # Tree data will be None if no trees were found
                if tree_data is not None:
                    tree_positions = tree_data['positions']
                    widths = tree_data['widths']
                    classes = tree_data['classes']

                    # Create the message to publish
                    data_all = TreeInfoMulti()
                    for width, position, classification in zip(widths, tree_positions, classes):
                        data_tree = TreeInfo()
                        position = TreePosition(x=position[0], y=position[1])
                        data_tree.position = position
                        data_tree.width = width
                        data_tree.classification = int(classification)
                        data_all.trees.append(data_tree)
                # If no trees, create a message with a single tree with a classification of 5. This is used to indicate
                # that no trees were found.
                else:
                    data_all = TreeInfoMulti()
                    data_tree = TreeInfo()
                    data_tree.position = TreePosition(x=0, y=0)
                    data_tree.width = 0
                    data_tree.classification = int(5)
                    data_all.trees.append(data_tree)
                data_all.header.stamp = time_stamp
                self.data_pub.publish(data_all)
                self.img_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))

    def shutdown_hook(self):

        # ... It doesn't like to be shut down. I think it's because of the multiprocessing. I tried to fix it and it
        # works better now, but it still doesn't always shut down properly. At the end of the script there's a line
        # that can be uncommented that seems to reliably kill it. It's not ideal according to the internet, but
        # I haven't had any problems with it.
        rospy.loginfo("Shutting down tree data node")

        self.image_queue.put((None, None, True), block=False)
        self.results_queue.put((None, None, True), block=False)

        rospy.sleep(1)

        self.inference_process.join()
        self.analyzer_process.join()
        self.error_queue.close()
        self.results_queue.close()
        self.image_queue.close()
        self.msg_to_pub_queue.close()
        self.analyzed_results_queue.close()
        self.error_queue.join_thread()
        self.results_queue.join_thread()
        self.image_queue.join_thread()
        self.msg_to_pub_queue.join_thread()
        self.analyzed_results_queue.join_thread()

        if self.inference_process.is_alive():
            rospy.logerr("Inference process failed to terminate")

        if self.analyzer_process.is_alive():
            rospy.logerr("Analyzer process failed to terminate")

        rospy.loginfo("Shutting down complete")
        sys.exit(0)

    def check_children_status_and_errors(self):
        # Check if the processes have sent any errors, if so, shut down the node (or at least try to...)
        num_errs = self.error_queue.qsize()
        if num_errs > 0:
            error_msg = self.error_queue.get(block=False)
            rospy.logerr(error_msg)
            self.image_sub.unregister()
            self.depth_sub.unregister()
            rospy.signal_shutdown(error_msg)

def process_results(results_queue, analyzed_results_queue, error_queue, x_offset, y_offset):
    # Process to analyze the results from the inference process, basically just sends them through the trunk analyzer
    # which does some filtering and then sends them back to the inference process for reasons explained below

    # Wrapped in a try except so that it if it crashes it can hopefully signal the main process to shut down
    try:
        trunk_analyzer = TrunkAnalyzer()

        while True:
            # This try except is to catch the timeout on that first line. I'm not sure the timeout is necessary, but
            # it hopefully avoids the process getting stuck in the queue.get() call
            try:
                # Get the results from the inference process, blocks for 1 second then passes
                result_dict, depth_image, shutdown = results_queue.get(timeout=1)

                # Just more stuff to try and make shutdown work, when a shutdown is signaled shutdown will be True
                if shutdown:
                    return

                # Send the results to the trunk analyzer
                tree_positions, widths, classes, img_x_positions, results_kept = trunk_analyzer.pf_helper(result_dict,
                                                                                        depth_image)

                # # Do some further processing on the results
                # if tree_positions is not None:
                #     # Switch sign on x_pos and y_pos
                #     tree_positions[:, 0] = -tree_positions[:, 0]
                #     tree_positions[:, 1] = -tree_positions[:, 1]

                # Add the offset to the positions
                if tree_positions is not None:
                    tree_positions[:, 0] += x_offset
                    tree_positions[:, 1] += y_offset

                    # img_x_positions = abs(img_x_positions - 320)
                    # # Values obtained from calibrate_widths.py, sept = True, poly = False
                    # widths = -0.006246 + (-2.0884248893265422e-05 * img_x_positions) + (1.057907234666699 * widths)

                    tree_data = {'positions': tree_positions, 'widths': widths, 'classes': classes, 'results_kept': results_kept}

                else:
                    tree_data = None

                # Send the results back to the inference process. Block false means that if the queue is full it will
                # just skip this and move on to the next iteration of the loop
                analyzed_results_queue.put(tree_data, block=False)

            except mp.queues.Empty:
                pass

    except Exception as e:
        error_msg = str(e)
        print(f"Error in analyzer subprocess of trunk data process: {error_msg}")
        error_queue.put(error_msg, block=False)

def run_inference(image_queue, results_queue, analyzed_results_queue, msg_to_pub_queue, error_queue):
    # Process to run the inference, basically just sends the images through the trunk segmenter and then sends the
    # results to the analyzer process. Also recieves the results back from the analyzer process so that it can use
    # the .plot() method to get a segmented image to publish. This is mostly because the .plot() method is pretty
    # efficient and i didn't want to try to implement it myself. Also, results can't seem to be sent through the
    # queue.

    # Wrapped in a try except so that it if it crashes it can hopefully signal the main process to shut down
    try:
        trunk_segmenter = TrunkSegmenter()
        results_list = []

        while True:

            # Make the segmented images to publish for all the results in the queue
            num_items = analyzed_results_queue.qsize()

            for i in range(num_items):
                tree_data = analyzed_results_queue.get(block=False)

                # Get the oldest results from the list
                results_full = results_list.pop(0)

                # If no trees were found, just use the original image
                if tree_data is None:
                    img = results_full.orig_img
                else:
                    # Otherwise, use the .plot() method to get the segmented image
                    results_kept = tree_data['results_kept']
                    results_full = results_full[results_kept]
                    img = results_full.plot()

                # Send the image and tree data to the main process to be published. I had some trouble with ros nodes
                # in a child process, but I may have just been doing something wrong. This works though.
                msg_to_pub_queue.put((img, tree_data), block=False)

            # This try except is to catch the timeout on that first line. I'm not sure the timeout is necessary, but
            # it hopefully avoids the process getting stuck in the queue.get() call
            try:
                # Get the images from the image queue, blocks for 1 second then passes, this is the only place that
                # blocks in this process
                rgb_image, depth_image, shutdown = image_queue.get(timeout=1)

                # This will be true if the main process has signaled a shutdown
                if shutdown:
                    return

                # Run the inference and save the results
                result_dict, result_full = trunk_segmenter.get_results(rgb_image)
                results_list.append(result_full)

                # Send the results to the analyzer process.
                results_queue.put((result_dict, depth_image, False), block=False)
            except mp.queues.Empty:
                pass

    except Exception as e:
        error_msg = str(e)
        print(f"Error in inference subprocess of trunk data process: {error_msg}")
        error_queue.put(error_msg, block=False)

if __name__ == '__main__':
    # Set the multiprocessing start method to spawn, as opposed the default, which is fork
    mp.set_start_method('spawn')
    tree_info_publisher = TreeInfoPublisher()

    # Run the main loop, just checks if there are new messages to publish in the queue or errors to handle
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
        tree_info_publisher.publish_data()
        tree_info_publisher.check_children_status_and_errors()

    # This is the line that can be uncommented to force the node to shut down, it seems to work at least most of the
    # time, but it's not ideal according to the internet
    # os.kill(os.getpid(), signal.SIGKILL)
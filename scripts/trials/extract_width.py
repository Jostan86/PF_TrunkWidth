from cv_bridge import CvBridge, CvBridgeError
import rosbag
from width_estimation import TrunkAnalyzer
import time
import pickle
import os
import cv2
import numpy as np

def pair_messages(d_msg, img_msg, bridge=CvBridge()):
    if d_msg is not None and img_msg is not None and d_msg.header.stamp == img_msg.header.stamp:
        try:
            depth_image = bridge.imgmsg_to_cv2(d_msg, "passthrough")
            color_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")

            # Check if color image height and width are divisible by 32
            if color_img.shape[0] % 32 != 0:
                color_img = color_img[:-(color_img.shape[0] % 32), :, :]
                depth_image = depth_image[:-(depth_image.shape[0] % 32), :]
            if color_img.shape[1] % 32 != 0:
                color_img = color_img[:, :-(color_img.shape[1] % 32), :]
                depth_image = depth_image[:, :-(depth_image.shape[1] % 32)]
        except CvBridgeError as e:
            print(e)
        return (depth_image, color_img)
    else:
        return None
def open_bag_file(bag_file_name, ):

    bag_file_dir = "/media/jostan/MOAD/research_data/2023_orchard_data/uncompressed/synced/pcl_mod/"
    cur_bag_file_name = bag_file_name
    topics = ["/registered/rgb/image", "/registered/depth/image"]



    path = bag_file_dir + cur_bag_file_name

    bag_data = rosbag.Bag(path)

    # Print unique topic names in bag_data file
    print(bag_data.get_type_and_topic_info()[1].keys())

    depth_msg = None
    color_msg = None
    paired_imgs = []
    time_stamps_img = []

    for topic, msg, t in bag_data.read_messages(topics=topics):

        if topic == topics[1]:
            depth_msg = msg
            paired_img = pair_messages(depth_msg, color_msg)
            if paired_img is not None:
                paired_imgs.append(paired_img)
                time_stamps_img.append(msg.header.stamp.to_sec())

        elif topic == topics[0]:
            color_msg = msg
            paired_img = pair_messages(depth_msg, color_msg)
            if paired_img is not None:
                paired_imgs.append(paired_img)
                time_stamps_img.append(msg.header.stamp.to_sec())

    return paired_imgs, time_stamps_img

if __name__ == "__main__":
    bag_file_dir = "/media/jostan/MOAD/research_data/2023_orchard_data/uncompressed/synced/pcl_mod/"
    data_save_dir = "/data/width_data/"

    bag_file_names = os.listdir(bag_file_dir)
    bag_file_names.sort()


    bag_file_name = "envy-trunks-01_4_converted_synced_pcl-mod.bag"

    # for bag_file_name in bag_file_names[0:20]:

    paired_imgs, time_stamps_img = open_bag_file(bag_file_name)

    trunk_analyzer = TrunkAnalyzer()


    widths_all = []
    times = []
    start = True
    for paired_img in paired_imgs:
        # paired_img = paired_imgs[0]

        time_start = time.time()
        tree_positions, widths, classes, img_seg, img_x_positions = trunk_analyzer.pf_helper(
            paired_img[1],
            paired_img[0],
            show_seg=True)
        time_end = time.time()
        if start:
            start = False
        else:
            times.append(time_end - time_start)
        # cv2.imshow("img_seg", img_seg)

        if widths is not None:
            widths_all.append(widths.tolist())
        else:
            widths_all.append(None)

    times = np.array(times)
    print("Average time: ", np.mean(times))

    # run = bag_file_name[12:16]
    # # Save widths to pickle file
    # with open(data_save_dir + "new_" + run + ".pkl", "wb") as f:
    #     pickle.dump(widths_all, f)




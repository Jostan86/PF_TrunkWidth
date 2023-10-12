import rosbag
import numpy as np


# path_to_bag = "/media/jostan/portabits/sept6/row106_107_sept.bag"
path_to_bag = "/media/jostan/MOAD/research_data/achyut_data/oct4/row_103_104.bag"
bag = rosbag.Bag(path_to_bag)

time_diffs_odom = []
time_diffs_img = []
for topic, msg, t in bag.read_messages(topics=['/odometry/filtered', '/camera/color/image_raw']):
    if topic == "/odometry/filtered":
        time_diffs_odom.append(msg.header.stamp.to_sec() - t.to_sec())
    if topic == "/camera/color/image_raw":
        time_diffs_img.append(msg.header.stamp.to_sec() - t.to_sec())

time_diffs = np.array(time_diffs_odom)
time_diffs_img = np.array(time_diffs_img)

print("Odom")
print("Mean: ", np.mean(time_diffs))
print("Std: ", np.std(time_diffs))
print("Max: ", np.max(time_diffs))
print("Min: ", np.min(time_diffs))

print("Image")
print("Mean: ", np.mean(time_diffs_img))
print("Std: ", np.std(time_diffs_img))
print("Max: ", np.max(time_diffs_img))
print("Min: ", np.min(time_diffs_img))

bag.close()
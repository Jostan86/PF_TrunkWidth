#!/usr/bin/env python

import rosbag
import os
import sys
import rospy


def split_rosbag(input_bag_path, chunk_duration):
    """Split a ROS bag into chunks of a specified duration.

    Args:
        input_bag_path (str): Path to the input ROS bag.
        chunk_duration (float): Duration of each chunk in seconds.
    """

    # Ensure the input file exists
    if not os.path.exists(input_bag_path):
        print(f"Error: {input_bag_path} does not exist.")
        return

    with rosbag.Bag(input_bag_path, 'r') as bag:
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        current_time = start_time

        chunk_count = 0

        while current_time < end_time:
            chunk_count += 1
            chunk_end = current_time + chunk_duration
            output_bag_path = f"{input_bag_path[:-4]}_chunk_{chunk_count}.bag"

            with rosbag.Bag(output_bag_path, 'w') as outbag:
                for topic, msg, t in bag.read_messages(start_time=rospy.Time(current_time),
                                                       end_time=rospy.Time(chunk_end)):
                    outbag.write(topic, msg, t)

            current_time += chunk_duration
            print(f"Wrote {output_bag_path}")

    print("All chunks have been generated.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} path_to_input_bag chunk_duration")
        sys.exit(1)

    input_bag_path = sys.argv[1]
    chunk_duration = float(sys.argv[2])

    split_rosbag(input_bag_path, chunk_duration)



import numpy as np

sorted_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
prev_img_time = 2.5
img_time = 5.5
start_index = next((i for i, v in enumerate(sorted_list) if v >= prev_img_time), 0)
end_index = next((i for i, v in enumerate(sorted_list) if v >= img_time), len(sorted_list))
odom_msgs_cur = sorted_list[start_index:end_index]

print(odom_msgs_cur)

# # Find the index for the start value
# start_index = next((i for i, v in enumerate(sorted_list) if v >= 4), None)
#
# # Find the index for the end value
# end_index = next((i for i, v in enumerate(sorted_list) if v > 10), len)
#
# print(start_index)  # This will output: 3
# print(end_index)  # This will output: 10
#
# # Slice the list between start and end indices
# filtered_list = sorted_list[start_index:end_index]
#
# print(filtered_list)  # This will output: [4, 5, 6, 7, 8, 9, 10]


x = [1]
print(x[:-1])

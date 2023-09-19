import numpy as np
import pickle
import os

data_save_dir = "/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/pf_trunk_width/data/width_data/"
diffs = []

file_names = os.listdir(data_save_dir)
file_names.sort()

# New filenames start with "new_"
filenames_new = [file_name for file_name in file_names if file_name.startswith("new_")]
filenames_old = [file_name for file_name in file_names if file_name.startswith("old_")]

for filename_new, filename_old in zip(filenames_new, filenames_old):

    # Load widths from pickle file
    widths_new = pickle.load(open(data_save_dir + filename_new, "rb"))
    widths_old = pickle.load(open(data_save_dir + filename_old, "rb"))


    for i, (old, new) in enumerate(zip(widths_old, widths_new)):
        if old is None or new is None:
            if old is None and new is None:
                pass
            else:
                print("i: ", i)
                print("old: ", old)
                print("new: ", new)
                print("")
        else:
            if len(old) == len(new):
                for old_width, new_width in zip(old, new):
                    diff = old_width - new_width
                    diffs.append(diff)
            else:
                print("i: ", i)
                print("diff num widhts")

diffs = np.array(diffs)
print("mean diff: ", np.mean(diffs))

print("top 10 diffs: ", diffs[np.argsort(diffs)[-10:]])

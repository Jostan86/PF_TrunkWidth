import numpy as np
import pickle
import os

data_save_dir = "/data/width_data/"
diffs = []

file_names = os.listdir(data_save_dir)
file_names.sort()

# New filenames start with "new_"
filenames_new = [file_name for file_name in file_names if file_name.startswith("new2_")]
filenames_old = [file_name for file_name in file_names if file_name.startswith("old_")]
sum_diifs = 0
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
                if len(old) > 1:
                    # rearrange widths so that they are in the same order, so minimize the total error
                    old_width = np.array(old)
                    new_width = np.array(new)
                    old = old_width[np.argsort(old_width)]
                    new = new_width[np.argsort(new_width)]

                for old_width, new_width in zip(old, new):
                    if abs(old_width - 0) < 0.00001:
                        continue
                    diff = abs(old_width - new_width)
                    sum_diifs += (old_width - new_width)
                    diffs.append(diff)
                    if diff > 0.01:
                        print(filename_new, "large i: ", i, diff)
            else:
                print(filename_new, "i: ", i)
                print("diff num widhts")

diffs = np.array(diffs)
print("mean diff: ", np.mean(diffs))

print("top 10 diffs: ", diffs[np.argsort(diffs)[-20:]])

print("sum diffs: ", sum_diifs)
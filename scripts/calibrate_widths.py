import json
import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# This script was/is used to come up with a way to correct the width estimate of the trunk depending on the x position
# of the trunk in the image. The idea is that the width estimate is more accurate when the trunk is in the center of
# the image, and less accurate when the trunk is near the edge of the image.
# Ground truth data for the test trees in the test block exist, and i saved a copy in the data directory of this package
# I then came up with a method in the bag_processor.py script to save the width estimate and the x position of the
# trunk in the image. I did this for data from sept collected by achyut with the warthog, data from feb collected
# with the husky, and I pulled all the saved estimations from the map data. I then used this script to test different
# methods of correcting the width estimate based on the x position of the trunk in the image. Really i just tested
# using a linear regression and a polynomial regression on the different data sets. After comparing the results, I
# decided to use a linear regrssion using the sept data. I decided to avoided polynomial regression because my data
# didnt have any post measurments, which are larger than the trees and weren't corrected well with the poly. As to
# why i used the sept data, it just gave about the same results as the feb data, and it's what we'll actually be using
# in the field, so it made sense.

map_data_path = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/orchard_data_analysis/data' \
                 '/2020_11_bag_data/afternoon2/tree_list_mod3.json'

# Load the tree data dictionary
with open(map_data_path, 'rb') as f:
    map_data = json.load(f)

# Get the tree data for the test trees
map_tree_data = {}
for tree in map_data:
    if tree['test_tree']:
        map_tree_info = {}
        test_tree_num = int(tree['test_tree_num'])
        map_tree_info['tree_num'] = tree['tree_num']
        map_tree_info['test_tree_num'] = test_tree_num
        map_tree_info['img_x_positions'] = tree['img_x_positions']
        map_tree_info['width_estimate'] = tree['width_estimate']
        map_tree_info['widths'] = tree['widths']

        map_tree_data[test_tree_num] = map_tree_info

# Get the ground truth data for the test trees
gt_tree_data = {}
gt_data_path = '../data/gt_diameter_data.csv'
# make the csv a pandas dataframe
gt_df = pd.read_csv(gt_data_path)
# cycle through the Tree Number and Diameter columns
for index, row in gt_df.iterrows():
    tree_num = int(row['Tree Num'])
    diameter = row['Diameter']
    # Divide by 100 to convert from cm to m
    gt_tree_data[tree_num] = diameter/100

# Get the tree data collected for the sept and feb data
# Get list of filenames with diameter data
dir_path = '../data/diam_data/'
filenames = os.listdir(dir_path)
sept_tree_data = {}
feb_tree_data = {}
# Loop through every file in the directory
for filename in filenames:
    with open(os.path.join(dir_path, filename), 'r') as f:
        tree_data = json.load(f)
    tree_num = int(tree_data['tree_num'])
    diameter = tree_data['width_estimates']
    img_x_positions = tree_data['img_x_positions']
    if filename[:3] == 'feb':
        feb_tree_data[tree_num] = {'widths': diameter, 'img_x_positions': img_x_positions}
    elif filename[:4] == 'sept':
        sept_tree_data[tree_num] = {'widths': diameter, 'img_x_positions': img_x_positions}

# Set some flags for which data to use this round to make the equation
use_feb = False
use_sept = True
use_map = False
# Set flag for whether to use a polynomial regression or a linear regression
polynomial = False

# Get the data for the linear regression
train_data = []
correct_data = []
# Get the tree numbers that data will be used from
if use_feb:
    tree_nums = [i for i in range(1514, 1532)]
if use_sept:
    tree_nums = [i for i in range(1514, 1524)]
    tree_nums.extend([i for i in range(1534, 1536)])
    tree_nums.extend([i for i in range(1537, 1542)])
    tree_nums.extend([i for i in range(1551, 1559)])
    tree_nums.extend([i for i in range(2534, 2541)])
if use_map:
    tree_nums = [i for i in range(1502, 1599)]
    tree_nums.extend([i for i in range(1201, 1300)])

for tree_num in tree_nums:
    # For the sept data i got data from the same trees twice for some of the trees, and i added 1000 to the num
    if use_sept and tree_num > 2000:
        tree_num -= 1000
        gt_width = gt_tree_data[tree_num]
        tree_num += 1000
    else:
        gt_width = gt_tree_data[tree_num]

    # Get the width estimates and x positions for the tree to use for training
    if use_feb:
        widths = feb_tree_data[tree_num]['widths']
        x_pos = feb_tree_data[tree_num]['img_x_positions']
    elif use_sept:
        widths = sept_tree_data[tree_num]['widths']
        x_pos = sept_tree_data[tree_num]['img_x_positions']
    elif use_map:
        widths = map_tree_data[tree_num]['widths']
        x_pos = map_tree_data[tree_num]['img_x_positions']
    # Correct the x positions to be relative to the center of the image and save the data to the training data lists
    for i in range(len(widths)):
        if use_feb or use_sept:
            # Feb and sept data were taken with a 640x480 camera
            x = abs(x_pos[i] - 320)
        elif use_map:
            # Map data was taken with a 848x480 camera
            x = abs(x_pos[i] - 424)
        width = widths[i]
        train_data.append([x, width])
        correct_data.append(gt_width)

print("Train data length:", len(train_data))

# Setup training
X = np.array(train_data)
y = np.array(correct_data)
if polynomial:
    # Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    # Train the model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Evaluate the model
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)
else:
    # Step 3: Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Step 4: Evaluate the model
    y_pred = model.predict(X)

# Print the equation
if not polynomial:
    b0 = model.intercept_
    b1, b2 = model.coef_
    print(f"The equation is: y = {b0} + {b1} * x1 + {b2} * x2")
else:
    # Extract the coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_
    print(f"The equation is: y = {intercept} + {coef[1]}*x1 + {coef[2]}*x2 + {coef[3]}*x1^2 + {coef[4]}*x1*x2 + {coef[5]}*x2^2")
print("Where x1 is the x position of the tree in pixels and x2 is the width of the tree in meters")
# Scatter plot of the true vs predicted widths
plt.scatter(X[:, 0], y, color='blue', label='True Width')
plt.scatter(X[:, 0], y_pred, color='red', label='Predicted Width')
plt.xlabel('X-Coordinate')
plt.ylabel('Width')
plt.legend()
plt.show()

# All the stuff below is just testing the model on each of the data sets and comparing the results. Basically i
# try to estimate the width of the trunk based on all the images for each tree, and compare the errors using
# the corrected widths for this calculation vs not correcting the widths. Well, at least for the sept, and feb data.
# For the map data, the 'old method' was actually to look at only the images where the trunk was near the center for
# the calculation. 
tree_nums = [i for i in range(1502, 1599)]
tree_nums2 = [i for i in range(1201, 1300)]
tree_nums.extend(tree_nums2)
new_width_estimates = []
old_width_estimates = []
for tree_num in tree_nums:
    gt_width = gt_tree_data[tree_num]
    map_widths = map_tree_data[tree_num]['widths']
    map_x_pos = map_tree_data[tree_num]['img_x_positions']
    tree_data = np.array([[abs(map_x_pos[i] - 424), map_widths[i]] for i in range(len(map_widths))])
    if polynomial:
        tree_data = poly.transform(tree_data)
    tree_widths = model.predict(tree_data)
    new_width_estimates.append(np.mean(tree_widths))
    old_width_estimates.append(map_tree_data[tree_num]['width_estimate'])

new_width_estimates = np.array(new_width_estimates)
old_width_estimates = np.array(old_width_estimates)
gt_widths = np.array([gt_tree_data[tree_num] for tree_num in tree_nums])

print('Map Tree Data')
# Calculate the mean absolute error
mae = np.mean(np.abs(gt_widths - new_width_estimates))
print("Mean Absolute Error New Estimate:", mae)

# Calculate the mean absolute error
mae = np.mean(np.abs(gt_widths - old_width_estimates))
print("Mean Absolute Error Old Estimate:", mae)

new_width_estimates = []
old_width_estimates = []
tree_nums = [i for i in range(1514, 1533)]
for tree_num in tree_nums:
    gt_width = gt_tree_data[tree_num]
    feb_widths = feb_tree_data[tree_num]['widths']
    feb_x_pos = feb_tree_data[tree_num]['img_x_positions']
    tree_data = np.array([[abs(feb_x_pos[i] - 320), feb_widths[i]] for i in range(len(feb_widths))])
    if polynomial:
        tree_data = poly.transform(tree_data)
    tree_widths = model.predict(tree_data)
    new_width_estimates.append(np.mean(tree_widths))
    old_width_estimates.append(np.mean(np.array(feb_widths)))

new_width_estimates = np.array(new_width_estimates)
old_width_estimates = np.array(old_width_estimates)
gt_widths = np.array([gt_tree_data[tree_num] for tree_num in tree_nums])

print('Feb Tree Data')
# Calculate the mean absolute error
mae = np.mean(np.abs(gt_widths - new_width_estimates))
print("Mean Absolute Error New Estimate:", mae)

# Calculate the mean absolute error
mae = np.mean(np.abs(gt_widths - old_width_estimates))
print("Mean Absolute Error Old Estimate:", mae)


new_width_estimates = []
old_width_estimates = []
gt_widths = []
tree_nums = [i for i in range(1514, 1524)]
tree_nums.extend([i for i in range(1534, 1536)])
tree_nums.extend([i for i in range(1537, 1542)])
tree_nums.extend([i for i in range(1551, 1559)])
tree_nums.extend([i for i in range(2534, 2541)])

for tree_num in tree_nums:
    if tree_num > 2000:
        tree_num -= 1000
        gt_width = gt_tree_data[tree_num]
        tree_num += 1000
    else:
        gt_width = gt_tree_data[tree_num]
    sept_widths = sept_tree_data[tree_num]['widths']
    sept_x_pos = sept_tree_data[tree_num]['img_x_positions']
    tree_data = np.array([[abs(sept_x_pos[i] - 320), sept_widths[i]] for i in range(len(sept_widths))])
    if polynomial:
        tree_data = poly.transform(tree_data)
    tree_widths = model.predict(tree_data)
    new_width_estimates.append(np.mean(tree_widths))
    old_width_estimates.append(np.mean(np.array(sept_widths)))
    gt_widths.append(gt_width)

new_width_estimates = np.array(new_width_estimates)
old_width_estimates = np.array(old_width_estimates)
gt_widths = np.array(gt_widths)

print('Sept Tree Data')
# Calculate the mean absolute error
mae = np.mean(np.abs(gt_widths - new_width_estimates))
print("Mean Absolute Error New Estimate:", mae)

# Calculate the mean absolute error
mae = np.mean(np.abs(gt_widths - old_width_estimates))
print("Mean Absolute Error Old Estimate:", mae)

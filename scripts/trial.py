import numpy as np
from scipy.spatial import KDTree
import time

# Generate random trees
np.random.seed(42)
num_trees = 1000
trees = np.random.rand(num_trees, 2) * 100

# Generate a random point
random_point = np.random.rand(1, 2) * 100

# Method 1: Distance calculation
start_time = time.time()

# Calculate distances between trees and the random point
distances = np.linalg.norm(trees - random_point, axis=1)

# Find the index of the tree with the minimum distance
closest_tree_index = np.argmin(distances)

# Get the closest tree coordinates
closest_tree_coordinates = trees[closest_tree_index]

distance_calculation_time = time.time() - start_time

# Method 2: KD-tree
# Build a kd-tree from the tree coordinates
tree_kdtree = KDTree(trees)

start_time = time.time()

# Query the kd-tree to find the closest tree to the random point
distance, closest_tree_index = tree_kdtree.query(np.array([[10, 10], [20, 20]]))

# Get the closest tree coordinates
closest_tree_coordinates_kdtree = trees[closest_tree_index]

kd_tree_time = time.time() - start_time

print("Random point:", random_point)
print("Closest tree coordinates (Distance Calculation):", closest_tree_coordinates)
print("Closest tree coordinates (KD-tree):", closest_tree_coordinates_kdtree)
print("Time taken (Distance Calculation):", distance_calculation_time)
print("Time taken (KD-tree):", kd_tree_time)
#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import copy
from fastslam_funcs_temp import pi_2_pi, find_closest_point


def object_global_location(particle, x_l, y_l):
    """
    Calculates the object's location in the global frame.

    Parameters:
    particle (object): particle object containing a state attribute which holds the x, y, theta values of the
    particle in the map frame

    Returns:
    A numpy array with the x coordinate in the zeroth column, and y coordinates in the first column
    """

    # Calculate the transformation matrix from local to global frame
    c, s = np.cos(particle.state[2]), np.sin(particle.state[2])
    R = [[c, -s, particle.state[0]],
         [s, c, particle.state[1]],
         [0, 0, 1]]

    # Calculate the object's location in the global frame
    x_g_obj = R[0][0] * x_l + R[0][1] * y_l + R[0][2]
    y_g_obj = R[1][0] * x_l + R[1][1] * y_l + R[1][2]

    return np.vstack((x_g_obj, y_g_obj)).T

def get_particle_weight_localize(sensed_trees, widths_sensed, tree_coords_gt, widths_gt, match_thresh=0.05,
                                 width_threshold=1.5):
    """

    Parameters
    ----------
    sensed_trees : x y coordinates of trees sensed, relative to map
    widths_sensed : widths of sensed trees
    tree_coords_gt : x y coordinate of known tree locations
    widths_gt : widths of known trees
    match_thresh : threshold to match a seen tree to a landmark
    width_threshold : threshold to match a tree based on width

    Returns
    -------
    score : tree score
    """
    score = 0

    for i, (tree_coord, width) in enumerate(zip(sensed_trees, widths_sensed)):
        closest_point_idx, distance = find_closest_point(tree_coord, tree_coords_gt)

        if abs(width - widths_gt[closest_point_idx]) < width_threshold and distance < match_thresh:
            score += 1 - distance * (.75/match_thresh)




    return score
#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import copy

def pi_2_pi(angle: float):
    """

    Parameters
    ----------
    angle :

    Returns
    -------

    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def resampling_localize(particles_states, weights, max_num_particles=2000000, min_num_particles=100,
                        Neff_threshold_min=0.5, Neff_threshold_max=0.99, change_rate=2):
    # Normalize weights
    weights = weights / np.sum(weights)

    # Calculate the effective number of particles
    num_particles = particles_states.shape[0]
    Neff = 1 / np.sum(np.square(weights))

    # Calculate the new number of particles
    if Neff < Neff_threshold_min * num_particles:
        num_particles = int(change_rate * num_particles)
    elif Neff > Neff_threshold_max * num_particles:
        num_particles = int((1/change_rate) * num_particles)

    if num_particles < min_num_particles:
        num_particles = min_num_particles

    if num_particles > max_num_particles:
        num_particles = max_num_particles

    print("Neff: ", Neff)
    print("num_particles: ", num_particles)

    # Resample particles using low variance sampling
    step_size = np.random.uniform(0, 1 / num_particles)
    cur_weight = weights[0]
    idx_w = 0
    new_particles = np.zeros((num_particles, 3))
    for idx_m in range(num_particles):
        U = step_size + idx_m / num_particles
        while U > cur_weight:
            idx_w += 1
            cur_weight += weights[idx_w]
        new_particles[idx_m, :] = particles_states[idx_w, :]

    return new_particles

def object_global_location(particle_states : np.ndarray, tree_locs : np.ndarray) -> np.ndarray:
    """
    Calculates the object's location in the global frame.

    Parameters:
    ----------
    particle_states : np.ndarray
        An array of shape (n, 3) containing the states of the particles
    tree_locs : np.ndarray
        An array of shape (n, 2) containing the locations of the trees in the local frame


    Returns
     ----------
     np.ndarray
        A MxNx2 numpy array, with x and y coordinates for each tree relative to each particle. Here n is the number
        of particles and m is the number of trees.
    """

    # Calculate sin and cos of particle angles
    s = np.sin(particle_states[:, 2])
    c = np.cos(particle_states[:, 2])

    tree_glob = np.zeros((tree_locs.shape[0], particle_states.shape[0], 2))

    for i in range(len(tree_locs)):
        # Calculate x and y coordinates of trees in global frame
        tree_glob[i, :, 0] = particle_states[:, 0] + tree_locs[i, 0] * c + tree_locs[i, 1] * -s
        tree_glob[i, :, 1] = particle_states[:, 1] + tree_locs[i, 0] * s + tree_locs[i, 1] * c

    return tree_glob

def get_particle_weight_localize0(particle_states, sensed_tree_coords, widths_sensed, kd_tree_map,
                                 widths_map, match_thresh=0.7, width_threshold=0.025) -> np.ndarray:
    """

    Parameters
    ----------
    particle_states : np.ndarray
        An array of shape (n, 3) containing the states of the particles
    sensed_tree_coords : np.ndarray
        An array of shape (m, n, 2) containing the locations of the trees in the global frame for each particle. m is the
        number of trees and n is the number of particles.
    widths_sensed : np.ndarray
        An array containing the widths of the trees.
    kd_tree_map : scipy.spatial.KDTree
        A KDTree containing the locations of the trees on the map.
    widths_map : np.ndarray
        An array containing the widths of the trees on the map.
    match_thresh : float
        The maximum distance between a tree sensed by the particle and a tree on the map for the two to be considered
        a match.
    width_threshold : float
        The maximum difference between the width of a tree sensed by the particle and a tree on the map for the two to
        be considered a match. Unit is meters

    Returns
    -------
    np.ndarray
        An array of shape (n,) containing the weights of the particles.
    """
    # Get the starting time
    start_time = time.time()

    scores = np.ones(particle_states.shape[0], dtype=float) * 0.0001

    for i in range(len(sensed_tree_coords)):
        distances, idx = kd_tree_map.query(sensed_tree_coords[i, :, :])
        width_diffs = np.abs(widths_sensed[i] - widths_map[idx])
        scores += ((1 - distances * (1/match_thresh)) ** 2) * (distances < match_thresh) * (width_diffs <
                                                                                            width_threshold)
        scores += (((1 - distances * (1/width_threshold)) ** 2) * (distances < width_threshold) * (distances <
                                                                                                   match_thresh))

    print("Time to calculate particle weights: ", time.time() - start_time)
    return scores


def get_particle_weight_localize(particle_states, sensed_tree_coords, widths_sensed, kd_tree_map,
                                 widths_map, match_thresh=0.7, width_threshold=0.025) -> np.ndarray:
    """

    Parameters
    ----------
    particle_states : np.ndarray
        An array of shape (n, 3) containing the states of the particles
    sensed_tree_coords : np.ndarray
        An array of shape (m, n, 2) containing the locations of the trees in the global frame for each particle. m is the
        number of trees and n is the number of particles.
    widths_sensed : np.ndarray
        An array containing the widths of the trees.
    kd_tree_map : scipy.spatial.KDTree
        A KDTree containing the locations of the trees on the map.
    widths_map : np.ndarray
        An array containing the widths of the trees on the map.
    match_thresh : float
        The maximum distance between a tree sensed by the particle and a tree on the map for the two to be considered
        a match.
    width_threshold : float
        The maximum difference between the width of a tree sensed by the particle and a tree on the map for the two to
        be considered a match. Unit is meters

    Returns
    -------
    np.ndarray
        An array of shape (n,) containing the weights of the particles.
    """

    dist_sd = 0.35
    width_sd = 0.025

    # Get the starting time
    start_time = time.time()

    scores = np.ones(particle_states.shape[0], dtype=float)
    print("Widths:", widths_sensed)

    for i in range(len(sensed_tree_coords)):

        distances, idx = kd_tree_map.query(sensed_tree_coords[i, :, :])
        width_diffs = np.abs((widths_sensed[i]-0.01) - (widths_map[idx]))

        prob_dist = probability_of_values(distances, 0, dist_sd)
        prob_width = probability_of_values(width_diffs, 0, width_sd)

        scores *= prob_dist * prob_width

    print("Time to calculate particle weights: ", time.time() - start_time)
    return scores

def probability_of_values(arr, mean, std_dev):
    norm_pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(arr - mean) ** 2 / (2 * std_dev ** 2))
    return norm_pdf

if __name__ == "__main__":
    R = np.diag([.1, np.deg2rad(2.0)]) ** 2
    # Initialize the particles as a numpy array of shape (num_particles, 3) with columns (x, y, theta)
    particles = np.zeros((1000, 3))

    np.random.seed(42)

    # Initialize the particles with a uniform distribution around the start pose
    particles[:, 0] = np.random.uniform(5, 15, 1000)
    particles[:, 1] = np.random.uniform(5, 15, 1000)
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, 1000)


    # u = np.array([[0.5], [2]])
    # dt = .1
    #
    # x2 = predict_particles(particles, u, dt, R)
    #
    # print(x2)
    tree_loc = np.array([[-1, 1], [0, 1], [0.5, 0.8], [10, 10]])
    tree_pos_rel_particles = object_global_location(particles, tree_loc)


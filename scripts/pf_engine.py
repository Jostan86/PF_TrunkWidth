#!/usr/bin/env python3

import os
import numpy as np
from scipy.spatial import KDTree
import time

class PFEngine:

    def __init__(self, map_data: dict, start_pose_center: iter, start_pose_height: float,
                 start_pose_width: float, num_particles: int, rotation: float, rand_seed: int=None) -> None:
        """

        Parameters
        ----------
        map_data
            Map data as a dictionary with keys 'positions', 'classes', and 'widths'
        start_pose_center
            Center of the start pose as a tuple (x, y), in meters from the origin of the map
        start_pose_height
            Spread of the particles along the y-axis, in meters
        start_pose_width
            Spread of the particles along the x-axis, in meters
        num_particles
            Number of particles to use at initialization
        rotation
            Rotation of the start pose, in radians
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)

        self.map_positions = map_data['positions']
        self.map_classes = map_data['classes']
        self.map_widths = map_data['widths']

        self.kd_tree = KDTree(self.map_positions)

        self.start_pose_center = np.array(start_pose_center)
        self.start_pose_width = start_pose_width
        self.start_pose_height = start_pose_height
        self.rotation = rotation

        self.particles = self.initialize_particles(num_particles)
        self.best_particle = self.particles[0]

        # Setup the covariances
        self.Q = np.diag([0.4, np.deg2rad(10.0)]) ** 2
        self.R = np.diag([.6, np.deg2rad(20.0)]) ** 2
        self.dist_sd = 0.35
        self.width_sd = 0.025

        self.Neff_threshold_min = 0.5
        self.Neff_threshold_max = 0.99
        self.change_rate = 2
        self.min_num_particles = 100
        self.max_num_particles = 2000000

        self.odom_zerod = False
        self.prev_t_odom = None
        self.prev_t_scan = None

        self.scan_msgs = []
        self.odom_msgs = []

        self.include_width = True

        self.scan_info_msgs = []




    def initialize_particles(self, num_particles: int) -> np.ndarray:
        """
        Initialize the particles for the particle filter

        Returns
        -------
        particles
            Particles as a numpy array of shape (num_particles, 3) with columns (x, y, theta)
        """

        # Initialize the particles as a numpy array of shape (num_particles, 3) with columns (x, y, theta)
        particles = np.zeros((num_particles, 3))

        # Initialize the particles with a uniform distribution around the start pose, in a rectangle
        # around the start pose center with width self.start_pose_width and height self.start_pose_height
        particles[:, 0] = np.random.uniform(self.start_pose_center[0] - self.start_pose_width / 2,
                                            self.start_pose_center[0] + self.start_pose_width / 2,
                                            num_particles)
        particles[:, 1] = np.random.uniform(self.start_pose_center[1] - self.start_pose_height / 2,
                                            self.start_pose_center[1] + self.start_pose_height / 2,
                                            num_particles)
        particles[:, 2] = np.random.uniform(0, 2 * np.pi, num_particles)
        # particles[:, 2] = np.random.uniform(np.deg2rad(230), np.deg2rad(250), num_particles)

        particles = self.rotate_around_point(particles, self.rotation, self.start_pose_center)



        return particles



    def save_odom_ros(self, odom_msg):
        if not self.odom_zerod:
            self.prev_t_odom = odom_msg.header.stamp.to_sec()
            self.odom_zerod = True
            return
        x_odom = odom_msg.twist.twist.linear.x
        theta_odom = odom_msg.twist.twist.angular.z
        dt_odom = odom_msg.header.stamp.to_sec() - self.prev_t_odom
        self.prev_t_odom = odom_msg.header.stamp.to_sec()
        self.odom_update(x_odom, theta_odom, dt_odom)

    def save_odom_loaded(self, x_odom, theta_odom, time_stamp):
        if not self.odom_zerod:
            self.prev_t_odom = time_stamp
            self.odom_zerod = True
            return
        dt_odom = time_stamp - self.prev_t_odom
        self.prev_t_odom = time_stamp
        self.odom_update(x_odom, theta_odom, dt_odom)


    def save_scan(self, scan_msg):
        self.scan_info_msgs = []
        self.scan_update(scan_msg)

    def odom_update(self, x_odom, theta_odom, dt_odom):

        u = np.array([[x_odom], [theta_odom]])
        self.motion_update(u, dt_odom)


    def scan_update(self, tree_msg):

        if tree_msg['positions'] is None:
            return

        postions_sense = np.array(tree_msg['positions'])
        classes_sense = np.array(tree_msg['classes'])
        widths_sense = np.array(tree_msg['widths'])

        # Remove trees that have a class of 2 (sprinklers)
        postions_sense = postions_sense[classes_sense != 2]
        widths_sense = widths_sense[classes_sense != 2]
        classes_sense = classes_sense[classes_sense != 2]

        tree_global_coords = self.object_global_location(self.particles, postions_sense)
        particles_weights = self.get_particle_weight_localize(self.particles, tree_global_coords, widths_sense,
                                                          self.kd_tree,
                                                         self.map_widths, match_thresh=0.50, width_threshold=.035, include_width=self.include_width)
        self.best_particle = self.particles[np.argmax(particles_weights)]
        self.particles = self.resampling_localize(self.particles, particles_weights)


    def motion_update(self, u: np.ndarray, dt: float):

        num_particles = self.particles.shape[0]

        # Make array of noise
        noise = np.random.randn(num_particles, 2) @ self.R
        ud = u + noise.T

        self.particles.T[0, :] += dt * ud[0, :] * np.cos(self.particles.T[2, :])
        self.particles.T[1, :] += dt * ud[0, :] * np.sin(self.particles.T[2, :])
        self.particles.T[2, :] += dt * ud[1, :]

        # angles1 = np.zeros((1, self.particles.T.shape[1]))
        # angles1[0, :] = self.particles.T[2, :]

        self.particles.T[2, :] = self.pi_2_pi(self.particles.T[2, :])

        self.best_particle[0] += dt * u[0] * np.cos(self.best_particle[2])
        self.best_particle[1] += dt * u[0] * np.sin(self.best_particle[2])
        self.best_particle[2] += dt * u[1]
        self.best_particle[2] = self.pi_2_pi(self.best_particle[2])




    def pi_2_pi(self, angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def resampling_localize(self, particles_states, weights):
        # Normalize weights
        weights = weights / np.sum(weights)

        # Calculate the effective number of particles
        num_particles = particles_states.shape[0]
        Neff = 1 / np.sum(np.square(weights))

        # Calculate the new number of particles
        if Neff < self.Neff_threshold_min * num_particles:
            num_particles = int(self.change_rate * num_particles)
        elif Neff > self.Neff_threshold_max * num_particles:
            num_particles = int((1 / self.change_rate) * num_particles)

        if num_particles < self.min_num_particles:
            num_particles = self.min_num_particles

        if num_particles > self.max_num_particles:
            num_particles = self.max_num_particles

        self.scan_info_msgs.append(("Neff: " + str(round(Neff, 3))))
        self.scan_info_msgs.append(("num_particles: " + str(num_particles)))

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

    def object_global_location(self, particle_states: np.ndarray, tree_locs: np.ndarray) -> np.ndarray:
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

    def get_particle_weight_localize(self, particle_states, sensed_tree_coords, widths_sensed, kd_tree_map,
                                     widths_map, match_thresh=0.7, width_threshold=0.025,
                                     include_width=True) -> np.ndarray:
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
        include_width : bool
            Whether to include the width of the trees in the weight calculation.

        Returns
        -------
        np.ndarray
            An array of shape (n,) containing the weights of the particles.
        """

        # Get the starting time
        start_time = time.time()

        scores = np.ones(particle_states.shape[0], dtype=float)
        # print("Widths:", widths_sensed)

        for i in range(len(sensed_tree_coords)):

            distances, idx = kd_tree_map.query(sensed_tree_coords[i, :, :])
            width_diffs = np.abs((widths_sensed[i] - 0.01) - (widths_map[idx]))

            prob_dist = self.probability_of_values(distances, 0, self.dist_sd)

            scores *= prob_dist

            if include_width:
                prob_width = self.probability_of_values(width_diffs, 0, self.width_sd)
                scores *= prob_width

        self.scan_info_msgs.append(("Time to calculate particle weights: " + str(round((time.time() - start_time) *
                                                                                       1000, 1)) + " ms"))
        return scores

    def probability_of_values(self, arr, mean, std_dev):
        norm_pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(arr - mean) ** 2 / (2 * std_dev ** 2))
        return norm_pdf

    def rotate_around_point(self, particles, angle_rad, center_point):
        """
        Rotate numpy array points (in the first two columns)
        around a given point by a given angle in degrees.

        Parameters:
        - matrix: numpy array where first two columns are x and y coordinates
        - angle_degree: angle to rotate in degrees
        - point: tuple of (x, y) coordinates of rotation center

        Returns:
        - Rotated numpy array
        """

        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Extract x and y columns
        xy_coords = particles[:, 0:2]

        # Translate points to origin based on the provided point
        translated_points = xy_coords - center_point

        # Apply rotation
        rotated_points = np.dot(translated_points, rotation_matrix.T)

        # Translate points back to original place
        rotated_translated_points = rotated_points + center_point

        # Replace original x and y values in the matrix with the rotated values
        particles[:, 0:2] = rotated_translated_points

        return particles


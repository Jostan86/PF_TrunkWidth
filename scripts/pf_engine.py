#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from fastslam_funcs_temp2 import pi_2_pi, object_global_location, get_particle_weight_localize, resampling_localize
class PFEngine:

    def __init__(self, map_data: dict, start_pose_center: tuple, start_pose_radius: float, num_particles: int) \
            -> None:
        """

        Parameters
        ----------
        map_data
            Map data as a dictionary with keys 'positions', 'classes', and 'widths'
        start_pose_center
            Center of the start pose as a tuple (x, y), in meters from the origin of the map
        start_pose_radius
            Spread of the start pose as a float, in meters, gives a square to put the particles in
        num_particles
            Number of particles to use at initialization
        """

        self.map_positions = map_data['positions']
        self.map_classes = map_data['classes']
        self.map_widths = map_data['widths']

        self.kd_tree = KDTree(self.map_positions)

        self.start_pose_center = np.array(start_pose_center)
        self.start_pose_radius = start_pose_radius

        self.particles = self.initialize_particles(num_particles)

        # Setup the covariances
        self.Q = np.diag([0.4, np.deg2rad(10.0)]) ** 2
        self.R = np.diag([.6, np.deg2rad(20.0)]) ** 2

        self.odom_zerod = False
        self.prev_t_odom = None
        self.prev_t_scan = None

        self.scan_msgs = []
        self.odom_msgs = []


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

        # Initialize the particles with a uniform distribution around the start pose, in a circle
        # around the start pose center with radius self.start_pose_radius
        angles = np.random.uniform(0, 2 * np.pi, num_particles)
        ranges = np.random.uniform(0, 1, num_particles)
        ranges = np.sqrt(ranges) * self.start_pose_radius
        particles[:, 0] = self.start_pose_center[0] + ranges * np.cos(angles)
        particles[:, 1] = self.start_pose_center[1] + ranges * np.sin(angles)
        particles[:, 2] = np.random.uniform(0, 2 * np.pi, num_particles)

        return particles



    def save_odom(self, odom_msg):
        # self.odom_msgs.append(odom_msg)
        # self.update_localize()
        self.odom_update(odom_msg)

    def save_scan(self, scan_msg):

        # self.scan_msgs.append(scan_msg)
        # self.update_localize()
        self.scan_update(scan_msg)

    # def update_localize(self):
    #     if len(self.scan_msgs) > 1 and len(self.odom_msgs) > 10:
    #         if self.scan_msgs[0]['header'].stamp.to_sec() < self.odom_msgs[0].header.stamp.to_sec():
    #             self.scan_update(self.scan_msgs.pop(0))
    #         else:
    #             self.odom_update(self.odom_msgs.pop(0))

    def odom_update(self, odom_msg):

        if not self.odom_zerod:
            self.prev_t_odom = odom_msg.header.stamp.to_sec()
            self.odom_zerod = True
            return

        u = np.array([[odom_msg.twist.twist.linear.x], [odom_msg.twist.twist.angular.z]])
        dt = odom_msg.header.stamp.to_sec() - self.prev_t_odom
        self.prev_t_odom = odom_msg.header.stamp.to_sec()
        self.motion_update(u, dt)


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

        tree_global_coords = object_global_location(self.particles, postions_sense)
        particles_weights = get_particle_weight_localize(self.particles, tree_global_coords, widths_sense, self.kd_tree,
                                                         self.map_widths, match_thresh=0.50, width_threshold=.035)

        self.particles = resampling_localize(self.particles, particles_weights)


    def motion_update(self, u: np.ndarray, dt: float):

        num_particles = self.particles.shape[0]

        # Make array of noise
        noise = np.random.randn(num_particles, 2) @ self.R
        ud = u + noise.T

        self.particles.T[0, :] += dt * ud[0, :] * np.cos(self.particles.T[2, :])
        self.particles.T[1, :] += dt * ud[0, :] * np.sin(self.particles.T[2, :])
        self.particles.T[2, :] += dt * ud[1, :]

        angles1 = np.zeros((1, self.particles.T.shape[1]))
        angles1[0, :] = self.particles.T[2, :]

        self.particles.T[2, :] = pi_2_pi(self.particles.T[2, :])





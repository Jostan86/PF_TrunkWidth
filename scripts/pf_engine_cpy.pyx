#!/usr/bin/env python3

import numpy as np
cimport numpy as cnp
cnp.import_array()
from scipy.spatial import KDTree
from scipy.stats import norm

# Make this file with this command:
# python setup.py build_ext --inplace

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
        self.map_widths = map_data['widths']

        self.kd_tree = KDTree(self.map_positions)

        self.start_pose_center = np.array(start_pose_center)
        self.start_pose_width = start_pose_width
        self.start_pose_height = start_pose_height
        self.rotation = rotation

        self.particles = self.initialize_particles(num_particles)
        self.particle_weights = np.ones(num_particles) / num_particles
        self.best_particle = self.particles[0]

        # Setup the covariances
        self.Q = np.diag([0.4, np.deg2rad(10.0)]) ** 2
        self.R = np.diag([.6, np.deg2rad(20.0)]) ** 2
        self.dist_sd = 0.35
        self.width_sd = 0.025

        # Maximum allowable error in K-L distance.
        self.epsilon = 0.05
        # Desired confidence in the calculated number of particles.
        self.delta = 0.05
        # Size of the bins in each dimension x, y for bin_size (in m) and theta for bin_angle.
        self.bin_size = 0.2 #m
        self.bin_angle = np.deg2rad(5)

        self.min_num_particles = 100
        self.max_num_particles = 2000000

        self.odom_zerod = False
        self.prev_t_odom = None
        self.prev_t_scan = None

        self.scan_msgs = []
        self.odom_msgs = []

        self.include_width = True

        self.histogram = None





    # def initialize_particles(self, num_particles: int) -> np.ndarray:
    def initialize_particles(self, int num_particles) -> cnp.ndarray[double]:
        cdef:
            cnp.ndarray[double, ndim = 2] particles
            double start_pose_center_x = self.start_pose_center[0]
            double start_pose_center_y = self.start_pose_center[1]
            double start_pose_width_by_2 = self.start_pose_width / 2
            double start_pose_height_by_2 = self.start_pose_height / 2


        # Initialize the particles as a numpy array of shape (num_particles, 3) with columns (x, y, theta)
        particles = np.zeros((num_particles, 3))

        # Initialize the particles with a uniform distribution around the start pose
        particles[:, 0] = np.random.uniform(start_pose_center_x - start_pose_width_by_2,
                                            start_pose_center_x + start_pose_width_by_2,
                                            num_particles)
        particles[:, 1] = np.random.uniform(start_pose_center_y - start_pose_height_by_2,
                                            start_pose_center_y + start_pose_height_by_2,
                                            num_particles)
        particles[:, 2] = np.random.uniform(0, 2 * np.pi, num_particles)

        particles = self.rotate_around_point(particles, self.rotation, self.start_pose_center)

        return particles
                # """
        # Initialize the particles for the particle filter
        #
        # Returns
        # -------
        # particles
        #     Particles as a numpy array of shape (num_particles, 3) with columns (x, y, theta)
        # """
        #
        # # Initialize the particles as a numpy array of shape (num_particles, 3) with columns (x, y, theta)
        # particles = np.zeros((num_particles, 3))
        #
        # # Initialize the particles with a uniform distribution around the start pose, in a rectangle
        # # around the start pose center with width self.start_pose_width and height self.start_pose_height
        # particles[:, 0] = np.random.uniform(self.start_pose_center[0] - self.start_pose_width / 2,
        #                                     self.start_pose_center[0] + self.start_pose_width / 2,
        #                                     num_particles)
        # particles[:, 1] = np.random.uniform(self.start_pose_center[1] - self.start_pose_height / 2,
        #                                     self.start_pose_center[1] + self.start_pose_height / 2,
        #                                     num_particles)
        # particles[:, 2] = np.random.uniform(0, 2 * np.pi, num_particles)
        #
        # particles = self.rotate_around_point(particles, self.rotation, self.start_pose_center)
        #
        # return particles

    def save_odom(self, x_odom, theta_odom, time_stamp):
        """Handle the odom message. This will be called every time an odom message is received.

        Parameters
        ----------
        x_odom
            The linear velocity of the robot in the forward direction, in meters per second
        theta_odom
            The angular velocity of the robot, in radians per second
        time_stamp
            The time stamp of the odom message, in seconds, i think it's utc time doesn't really matter since this
            program zeros the time stamp on the first odom message
            """
        # If this is the first odom message, zero the time and return
        if not self.odom_zerod:
            self.prev_t_odom = time_stamp
            self.odom_zerod = True
            return
        dt_odom = time_stamp - self.prev_t_odom
        self.prev_t_odom = time_stamp
        u = np.array([[x_odom], [theta_odom]])
        self.motion_update(u, dt_odom)

    def scan_update(self, tree_msg):

        if tree_msg['positions'] is None:
            return

        postions_sense = np.array(tree_msg['positions'])
        widths_sense = np.array(tree_msg['widths'])

        tree_global_coords = self.object_global_location(self.particles, postions_sense)
        self.particle_weights = self.get_particle_weight_localize(self.particles, tree_global_coords, widths_sense)

        # Normalize weights
        self.particle_weights /= np.sum(self.particle_weights)

        self.best_particle = self.particles[np.argmax(self.particle_weights)]

        self.resample_particles()


    def motion_update(self, u: np.ndarray, dt: float):
        """Propagate the particles forward in time using the motion model."""

        num_particles = self.particles.shape[0]

        # Make array of noise
        noise = np.random.randn(num_particles, 2) @ self.R
        # Add noise to control/odometry velocities
        ud = u + noise.T

        # Update particles based on control/odometry velocities and time step size
        self.particles.T[0, :] += dt * ud[0, :] * np.cos(self.particles.T[2, :])
        self.particles.T[1, :] += dt * ud[0, :] * np.sin(self.particles.T[2, :])
        self.particles.T[2, :] += dt * ud[1, :]

        # Wrap angles between -pi and pi
        self.particles.T[2, :] = (self.particles.T[2, :] + np.pi) % (2 * np.pi) - np.pi

        # Update best particle with raw odom velocities
        self.best_particle[0] += dt * u[0] * np.cos(self.best_particle[2])
        self.best_particle[1] += dt * u[0] * np.sin(self.best_particle[2])
        self.best_particle[2] += dt * u[1]
        self.best_particle[2] = (self.best_particle[2] + np.pi) % (2 * np.pi) - np.pi


    def resample_particles(self):
        """Resample the particles according to the particle weights."""

        num_particles = self.calculate_num_particles(self.particles)


        step_size = np.random.uniform(0, 1 / num_particles)
        cur_weight = self.particle_weights[0]
        idx_w = 0
        new_particles = np.zeros((num_particles, 3))
        for idx_m in range(num_particles):
            U = step_size + idx_m / num_particles
            while U > cur_weight:
                idx_w += 1
                cur_weight += self.particle_weights[idx_w]
            new_particles[idx_m, :] = self.particles[idx_w, :]

        self.particles = new_particles
        self.particle_weights = np.ones(num_particles) / num_particles

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

    def get_particle_weight_localize(self, particle_states, sensed_tree_coords, widths_sensed) -> np.ndarray:
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

        Returns
        -------
        np.ndarray
            An array of shape (n,) containing the weights of the particles.
        """


        scores = np.ones(particle_states.shape[0], dtype=float)
        # print("Widths:", widths_sensed)

        for i in range(len(sensed_tree_coords)):

            distances, idx = self.kd_tree.query(sensed_tree_coords[i, :, :])
            width_diffs = np.abs(widths_sensed[i] - (self.map_widths[idx]))

            prob_dist = self.probability_of_values(distances, 0, self.dist_sd)

            scores *= prob_dist

            if self.include_width:
                prob_width = self.probability_of_values(width_diffs, 0, self.width_sd)
                scores *= prob_width

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

    def calculate_num_particles(self, particles):
        """
        Calculate the number of particles to use based on KLD-sampling.

        Args:
        - particles (np.array): Array of shape (num_particles, 3) with columns (x, y, theta).
        - epsilon (float): Maximum allowable error in K-L distance.
        - delta (float): Desired confidence in the calculated number of particles.
        - bin_size (tuple): Size of the bins in each dimension (x, y, theta).

        Returns:
        - int: Number of particles for the next timestep.
        """

        # Create multi-dimensional grid
        x_bins = np.arange(particles[:, 0].min(), particles[:, 0].max() + self.bin_size, self.bin_size)
        y_bins = np.arange(particles[:, 1].min(), particles[:, 1].max() + self.bin_size, self.bin_size)
        theta_bins = np.arange(particles[:, 2].min(), particles[:, 2].max() + self.bin_angle, self.bin_angle)

        # Calculate histogram to determine number of non-empty bins (k)
        self.histogram, _ = np.histogramdd(particles, bins=(x_bins, y_bins, theta_bins))
        k = np.sum(self.histogram > 0)

        if k == 1:
            return self.min_num_particles



        if n < self.min_num_particles:
            n = self.min_num_particles

        if n > self.max_num_particles:
            n = self.max_num_particles

        # print("Number of particles:", int(np.ceil(n)))

        return int(np.ceil(n))
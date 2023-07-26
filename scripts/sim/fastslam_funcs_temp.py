#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import copy

def polar_euclidean_distances(distances: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    This function takes in an array of the distances and an array of the angles to a set of points, and returns the
    euclidean distance between consecutive points. The two arrays should be the same size
    Parameters
    ----------
    distances : 1D array of ranges / distances
    angles : 1D array of bearings / angles to points

    Returns
    -------
    1D array of distances between consecitive points, will be 1 shorter than input arrays
    """
    # Convert polar coordinates to Cartesian coordinates using polar_to_cartesian function
    coords = polar_to_cartesian(distances, angles)
    # Compute pairwise differences in x and y coordinates
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    # Compute Euclidean distances between points
    dists = np.sqrt(dx**2 + dy**2)
    return dists


def polar_to_cartesian(distances: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    This function takes in a numpy array of distances, and a numpy array of angles, and returns a 2D array of the
    x y coordinates in the cartesian frame.
    Parameters
    ----------
    distances : 1D array of distances / ranges to points
    angles : 1D array of angles / bearings to points

    Returns
    -------
    x y cartesian coordinates of points
    """
    # Convert polar coordinates to Cartesian coordinates
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    # Combine x and y coordinates into a 2D numpy array
    coords = np.vstack((x, y)).T
    return coords


def rotate_coords(coords: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates a 2D numpy array of Cartesian coordinates counterclockwise around
    the origin by the specified angle (in radians).
    Parameters
    ----------
    coords : A 2D numpy array of Cartesian coordinates, where
        each row represents a point in 2D space.
    angle : The angle (in radians) by which to rotate the coordinates.

    Returns
    -------
        np.ndarray: A 2D numpy array of Cartesian coordinates representing the
            rotated points.
    """
    # Define the rotation matrix
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Rotate the coordinates around the origin
    rotated_coords = np.dot(coords, rot_matrix)
    return rotated_coords


def mask_large_values(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns a boolean mask that filters out values in the input array
    that are larger than the specified threshold.

    Parameters:
        arr (np.ndarray): The input array to be masked.
        threshold (float): The threshold value above which elements
            in the input array should be filtered out.

    Returns:
        np.ndarray: A boolean mask that can be used to index the input
            array and filter out elements larger than the specified threshold.
    """
    mask = arr <= threshold
    return mask

def group_points(distances: np.ndarray, threshold: float) -> np.ndarray:
    """
    Makes clusters based on distances between consecutive points

     Parameters:
        distances (np.ndarray): A numpy array where each values is the distance between 2 consecutive points in.
        threshold (float): The distance threshold for starting a new cluster.

    Returns:
        A numpy array of integers representing the cluster index for each point.
    """

    curr_cluster = 0
    cluster_assignment = [curr_cluster]
    for distance in distances:
        if distance > threshold:
            curr_cluster += 1
            cluster_assignment.append(curr_cluster)
        else:
            cluster_assignment.append(curr_cluster)
    return np.array(cluster_assignment)

def make_groups(points: np.ndarray, group_assignments: np.ndarray) -> List[np.ndarray]:
    """
    Group the 2D points in `points` based on their group assignments in `group_assignments`.

    Parameters:
    -----------
    points : np.ndarray
        A 2D NumPy array of shape (n, 2) containing the x and y coordinates of `n` points.
    group_assignments : np.ndarray
        A 1D NumPy array of length `n` containing the group assignments for each point.

    Returns:
    --------
    List of np.ndarray
        A list of NumPy arrays, one for each group, containing the 2D points assigned to that group.
    """
    # Get the number of unique group assignments
    num_groups = len(set(group_assignments))

    # Initialize a list to hold the points for each group
    groups = [np.empty((0, 2)) for _ in range(num_groups)]

    # Iterate over each point and group assignment
    for i, group in enumerate(group_assignments):
        # Add the current point to the corresponding group
        groups[group] = np.vstack((groups[group], points[i]))

    return groups

def plot_coords(coords: np.ndarray, size: float = 3) -> None:
    """
    Plots a scatter plot of 2D coordinates after rotating them by 90 degrees.

    Parameters:
    -----------
    coords : np.ndarray
        The 2D array of coordinates to plot.
    size : float, optional (default=3)
        The size of the plot in both the x and y directions.

    Returns:
    --------
    None
        The function plots the scatter plot but does not return anything.

    """
    # Rotate the coordinates around the origin
    rotated_coords = rotate_coords(coords, 1.5 * np.pi)
    plt.scatter(rotated_coords[:, 0], rotated_coords[:, 1], s=5)

    plot_size = size
    # Set axis limits and aspect ratio
    plt.axis('equal')
    plt.xlim(-plot_size, plot_size)
    plt.ylim(-1, plot_size)

    # Add labels and title
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('My Scatter Plot')

    # Display the plot
    plt.show()

def plot_coord_groups(grouped_points: List[np.ndarray], size: float = 3, animate: bool = False) -> None:
    """
    Plots a scatter plot of groups of 2D coordinates after rotating them by 90 degrees, with each group a different
    color. Can also animate, although it seems to sometimes break when animating

    Parameters:
    -----------
    grouped_points : List of np.ndarray
        The list of groups of coordinates to plot, each group being a 2D array.
    size : float, optional (default=3)
        The size of the plot in both the x and y directions.
    animate: bool, optional (default=False)

    Returns:
    --------
    None
        The function plots the scatter plot but does not return anything.

    """

    plot_size = size

    for point_group in grouped_points:
        # Rotate the coordinates around the origin
        rotated_point_group = rotate_coords(point_group, 1.5 * np.pi)

        # Set axis limits and aspect ratio
        plt.axis('equal')
        plt.xlim(-plot_size, plot_size)
        plt.ylim(-plot_size, plot_size)

        # Add labels and title
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt.title('Groups')

        if animate:
            # Plot all the groups
            for point_group_full in grouped_points:
                rotated_point_group_full = rotate_coords(point_group_full, 1.5 * np.pi)
                plt.scatter(rotated_point_group_full[:, 0], rotated_point_group_full[:, 1], s=5, c='b')

            # Plot just the one group
            plt.scatter(rotated_point_group[:, 0], rotated_point_group[:, 1], s=5, c='r')
            time.sleep(.5)

            # Display the plot
            plt.show()
        else:
            # Plot the group
            plt.scatter(rotated_point_group[:, 0], rotated_point_group[:, 1], s=5)
    plt.show()

class Particle:

    def __init__(self, num_particle: int, num_lm: int, lm_size=2):
        """
        Stores particle info
        @param num_particle: Total number of particles
        @param num_lm: Number of landmarks
        @param lm_size: State size of landmarks (should just be (x y)
        """
        self.w = 1.0 / num_particle

        # State in the form x, y, theta
        self.state = np.array([[0.0], [0.0], [0.0]])
        # self.y = 0.0
        # self.yaw = 0.0
        # landmark x-y positions
        self.lm = np.zeros((num_lm, lm_size))
        # landmark position covariance
        self.lmP = np.zeros((num_lm * lm_size, lm_size))

class ParticleLocalize:

    def __init__(self, num_particle: int):
        """
        Stores particle state and weight
        @param num_particle: Total number of particles
        """
        self.w = 1.0 / num_particle

        # State in the form x, y, theta
        self.state = np.array([[0.0], [0.0], [0.0]])



def motion_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Adds motion to a particle
    Parameters
    ----------
    x : current state - should be a vertical numpy array in the form [[x], [y], [theta]]
    u : motion model, in this case the linear and angular velocity in the form [[v], [w]], noise should already
    be applied if desired
    dt : time the partice has been moving for since the last update

    Returns
    -------
    updated state
    """

    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[dt * math.cos(x[2, 0]), 0],
                  [dt * math.sin(x[2, 0]), 0],
                  [0.0, dt]])
    x = F @ x + B @ u

    x[2, 0] = pi_2_pi(x[2, 0])
    return x

def pi_2_pi(angle: float):
    """

    Parameters
    ----------
    angle :

    Returns
    -------

    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def predict_particles(particles, u, dt, R):
    for particle in particles:
        px = particle.state
        ud = u + (np.random.randn(1, 2) @ R).T  # add noise
        px = motion_model(px, ud, dt)
        particle.state = px



def update_with_observation(particles, sensed_trees, Q, num_lm_filled, new_tree_thresh=0.05):
    # Find the particle with the current max weight
    max_prob_particle = 0
    for i, particle in enumerate(particles):
        if particle.w > particles[max_prob_particle].w:
            max_prob_particle = i

    max_prob_particle = copy.deepcopy(particles[max_prob_particle])

    xy_sensed = particle_rb_to_map_xy(max_prob_particle, sensed_trees[:2, :])
    xy_landmarks = max_prob_particle.lm.copy()

    for i, tree_coord in enumerate(xy_sensed):
        closest_point_idx, distance = find_closest_point(tree_coord, xy_landmarks)

        if distance > new_tree_thresh:
            z = np.array([sensed_trees[0, i], sensed_trees[1, i], num_lm_filled])
            for particle in particles:
                add_new_lm(particle, z, Q)
            num_lm_filled += 1

        else:
            for particle in particles:
                z = np.array([sensed_trees[0, i], sensed_trees[1, i], closest_point_idx])
                w = compute_weight(particle, z, Q)
                particle.w *= w
                update_landmark(particle, z, Q)

    return num_lm_filled

def get_particle_weight_localize(sensed_trees, landmarks, match_thresh=0.05):
    """

    Parameters
    ----------
    sensed_trees : x y coordinates of trees sensed, relative to particle
    landmarks : x y coordinate of known tree locations
    match_thresh : threshold to match a seen tree to a landmark

    Returns
    -------
    score : tree score
    """
    score = 0
    for i, tree_coord in enumerate(sensed_trees):


        closest_point_idx, distance = find_closest_point(tree_coord[:2], landmarks)

        if distance < match_thresh:
            score += 1 - distance * 15

    return score


def add_new_lm(particle, z, Q):

    # range to new landmark
    r = z[0]
    # angle to new landmark
    b = z[1]
    # ID of new landmark
    lm_id = int(z[2])

    # sin and cos of angle to landmark, in map frame
    s = math.sin(pi_2_pi(particle.state[2, 0] + b))
    c = math.cos(pi_2_pi(particle.state[2, 0] + b))

    # x and y location of landmark observed by particle, on map frame
    particle.lm[lm_id, 0] = particle.state[0, 0] + r * c
    particle.lm[lm_id, 1] = particle.state[1, 0] + r * s

    # covariance
    Gz = np.array([[c, -r * s],
                   [s, r * c]])

    particle.lmP[2 * lm_id:2 * lm_id + 2] = Gz @ Q @ Gz.T

def particle_rb_to_map_xy(particle, ranges_angles):
    """

    @param particle: Particle to use state of
    @param ranges_angles: 2d array with ranges in row 0, angles in row 1
    @return:
    """
    # sin and cos of angle to landmark, in map frame
    s = np.sin(pi_2_pi(particle.state[2] + ranges_angles[1, :]))
    c = np.cos(pi_2_pi(particle.state[2] + ranges_angles[1, :]))

    # x and y location of landmark observed by particle, on map frame
    x = particle.state[0] + ranges_angles[0, :] * c.T
    y = particle.state[1] + ranges_angles[0, :] * s.T

    return np.vstack((x, y)).T

def compute_weight(particle, z, Q):
    lm_id = int(z[2])

    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)
    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = pi_2_pi(dx[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print("singuler")
        return 1.0

    num = math.exp(-0.5 * dx.T @ invS @ dx)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))
    w = num / den

    return w

def compute_jacobians(particle, xf, Pf, Q):
    dx = xf[0, 0] - particle.state[0, 0]
    dy = xf[1, 0] - particle.state[1, 0]
    d2 = dx**2 + dy**2
    d = math.sqrt(d2)

    zp = np.array(
        [d, pi_2_pi(math.atan2(dy, dx) - particle.state[2, 0])]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q

    return zp, Hv, Hf, Sf

def update_landmark(particle, z, Q):

    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    xf, Pf = update_KF_with_cholesky(xf, Pf, dz, Q, Hf)

    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle

def update_KF_with_cholesky(xf, Pf, v, Q, Hf):
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q

    S = (S + S.T) * 0.5
    SChol = np.linalg.cholesky(S).T
    SCholInv = np.linalg.inv(SChol)
    W1 = PHt @ SCholInv
    W = W1 @ SCholInv.T

    x = xf + W @ v
    P = Pf - W1 @ W1.T

    return x, P

def get_tree_locations_from_scan(ranges: List, angle_min: float, angle_max: float, max_dist=2, group_thresh=0.08,
                                 min_scan_for_tree=3):
    """

    Parameters
    ----------
    ranges :
    angle_min :
    angle_max :
    max_dist :
    group_thresh :
    min_scan_for_tree :

    Returns
    -------

    """
    # Make array of the angles and ranges
    angles = np.linspace(angle_min, angle_max, len(ranges))
    ranges = np.array(ranges)

    # Mask to remove zeros
    mask = np.isclose(ranges, 0, rtol=1e-5, atol=1e-5)
    angles = angles[~mask]
    ranges = ranges[~mask]

    # Mask to remove large values
    mask = mask_large_values(ranges, max_dist)
    angles = angles[mask]
    ranges = ranges[mask]

    # Get the distances between consecutive points
    distances = polar_euclidean_distances(ranges, angles)

    # Get cluster assignments based on the distances between points, if they are closer than group thresh together
    # than they will be put in a group
    cluster_assignments = group_points(distances, group_thresh)

    # Make ranges and angles into 1 array
    # TODO: do this at beginning of function
    observed_trees = np.vstack((ranges, angles))

    # Group the trees, returns a list of arrays, each array holds a group
    grouped_points_rb = make_groups(observed_trees.T, cluster_assignments)

    # Remove groups with only one or two laser scans
    grouped_points_rb = [arr for arr in grouped_points_rb if len(arr) > (min_scan_for_tree - 1)]

    # Make array for location estimate of trees
    tree_loc_est = np.zeros((3, len(grouped_points_rb)))

    # Fill range and angle values with group means
    for i, group in enumerate(grouped_points_rb):
        tree_loc_est[0, i], tree_loc_est[1, i] = np.mean(group, axis=0)

    return tree_loc_est

def find_closest_point(point, points_array):
    """
    Finds the index of the point in a given 2D numpy array closest to a given point.

    Arguments:
    point -- a 1D numpy array of shape (2,) representing the point for which to find the closest point
    points_array -- a 2D numpy array of shape (n, 2) containing the points to search through

    Returns:
    An integer representing the index of the closest point to the input point in the input points_array,
    and a float representing the distance between the two points.
    """
    dists = np.linalg.norm(points_array - point, axis=1)  # Calculate the Euclidean distances between point and all points in points_array
    closest_point_idx = np.argmin(dists)  # Get the index of the point in points_array with the smallest distance to point
    distance = dists[closest_point_idx]  # Get the distance between the input point and the closest point
    return closest_point_idx, distance

def normalize_weight(particles):

    sumw = sum([p.w for p in particles])

    try:
        for particle in particles:
            particle.w /= sumw
    except ZeroDivisionError:
        for particle in particles:
            particle.w = 1.0 / len(particles)

def resampling(particles, NTH):
    """
    low variance re-sampling
    """

    normalize_weight(particles)

    num_particles = len(particles)

    weights = []

    # Get array of weights
    for particle in particles:
        weights.append(particle.w)
    weights = np.array(weights)

    # Calculate effective particle number
    Neff = 1.0 / (weights @ weights.T)

    # If Neff is above threshold, then resample
    if Neff < NTH:  # resampling
        # print("resampling")

        # Calculate cumulative weight
        weight_cumulative = np.cumsum(weights)
        base = np.cumsum(weights * 0.0 + 1 / num_particles) - 1 / num_particles
        resampleid = base + np.random.rand(base.shape[0]) / num_particles

        inds = []
        ind = 0
        for ip in range(num_particles):
            while ((ind < weight_cumulative.shape[0] - 1) and (resampleid[ip] > weight_cumulative[ind])):
                ind += 1
            inds.append(ind)

        tparticles = particles[:]
        for i in range(len(inds)):
            # particles[i].state = tparticles[inds[i]].state
            particles[i] = copy.deepcopy(tparticles[inds[i]])
            particles[i].w = 1.0 / num_particles

    # return particles, inds
#
def resampling_localize(particles):
    normalize_weight(particles)
    num_particles = len(particles)
    weights = []

    # Get array of weights
    for particle in particles:
        weights.append(particle.w)
    weights = np.array(weights)

    # Calculate cumulative weight
    weight_cumulative = np.cumsum(weights)
    base = np.cumsum(weights * 0.0 + 1 / num_particles) - 1 / num_particles
    resampleid = base + np.random.rand(base.shape[0]) / num_particles

    inds = []
    ind = 0
    for ip in range(num_particles):
        while ((ind < weight_cumulative.shape[0] - 1) and (resampleid[ip] > weight_cumulative[ind])):
            ind += 1
        inds.append(ind)

    tparticles = particles[:]
    for i in range(len(inds)):
        # particles[i].state = tparticles[inds[i]].state
        particles[i] = copy.deepcopy(tparticles[inds[i]])
        particles[i].w = 1.0 / num_particles


if __name__ == "__main__":
    R = np.diag([1.0, np.deg2rad(20.0)]) ** 2
    x1 = np.array([[1], [1], [np.pi/4]])
    print(x1[2])
    u = np.array([[1], [1]])
    dt = .1

    x2 = motion_model(x1, u, dt)

    print(x2)

import numpy as np
from scipy.stats import norm


def calculate_num_particles(particles, epsilon, delta, bin_size):
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
    x_bins = np.arange(particles[:, 0].min(), particles[:, 0].max() + bin_size[0], bin_size[0])
    y_bins = np.arange(particles[:, 1].min(), particles[:, 1].max() + bin_size[1], bin_size[1])
    theta_bins = np.arange(particles[:, 2].min(), particles[:, 2].max() + bin_size[2], bin_size[2])

    # Calculate histogram to determine number of non-empty bins (k)
    hist, _ = np.histogramdd(particles, bins=(x_bins, y_bins, theta_bins))
    k = np.sum(hist > 0)

    # Calculate z_1-delta (upper 1-delta quantile of the standard normal distribution)
    z_1_delta = norm.ppf(1 - delta)

    # Calculate n using the derived formula
    first_term = (k - 1) / (2 * epsilon)
    second_term = (1 - (2 / (9 * (k - 1))) + np.sqrt(2 * z_1_delta / (9 * (k - 1)))) ** 3
    n = first_term * second_term

    return int(np.ceil(n))


# Example usage
particles = np.array([[1, 2, 0.5], [2, 3, 0.7], [3, 1, 0.9]])
epsilon = 0.1
delta = 0.05
bin_size = (0.5, 0.5, 0.1)
print(calculate_num_particles(particles, epsilon, delta, bin_size))

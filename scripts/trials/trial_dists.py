import scipy.spatial
import numpy as np

# Make a 2x1000000 array of random numbers
a = np.random.rand(1000000, 2)
dists = scipy.spatial.distance.pdist(a)
print(np.min(a))
print(np.max(dists))

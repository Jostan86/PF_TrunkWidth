import numpy as np

R = np.diag([.6, np.deg2rad(20.0)]) ** 2
noise = np.random.randn(5, 2) @ R

print(R)
print(noise)
import numpy as np


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state_estimate = np.array(initial_state)  # [x_position, y_position, orientation]
        self.estimate_covariance = np.array(initial_covariance)  # Initial covariance matrix
        self.process_noise = np.array(process_noise)  # Process noise covariance
        self.measurement_noise = np.array(measurement_noise)  # Measurement noise covariance

    def process_update(self, odometry):
        # Update state estimate using odometry data (motion model)
        # Here, we assume a simple motion model where the robot can move in x and y and rotate.
        # You might need to adapt this to match your specific robot's motion model.

        delta_x, delta_y, delta_theta = odometry
        self.state_estimate[0] += delta_x
        self.state_estimate[1] += delta_y
        self.state_estimate[2] += delta_theta

        # Update the estimate covariance
        F = np.array([[1, 0, -delta_y], [0, 1, delta_x], [0, 0, 1]])  # Jacobian of the motion model
        self.estimate_covariance = np.dot(F, np.dot(self.estimate_covariance, F.T)) + self.process_noise

    def measurement_update(self, measurement):
        # Update state estimate using the measurement (best particle position and orientation)

        H = np.eye(3)  # Observation matrix
        y = measurement - np.dot(H, self.state_estimate)  # Measurement residual

        S = np.dot(H, np.dot(self.estimate_covariance, H.T)) + self.measurement_noise  # Residual covariance
        K = np.dot(self.estimate_covariance, np.dot(H.T, np.linalg.inv(S)))  # Kalman gain

        # Update the state estimate and covariance
        self.state_estimate += np.dot(K, y)
        self.estimate_covariance = self.estimate_covariance - np.dot(K, np.dot(H, self.estimate_covariance))


# Usage example
kf = KalmanFilter(initial_state=[0, 0, 0], initial_covariance=np.eye(3), process_noise=np.eye(3) * 0.1,
                  measurement_noise=np.eye(3) * 0.1)

# Update process (using odometry)
kf.process_update(odometry=[0.1, 0.1, 0.01])

# Update measurement (using best particle)
kf.measurement_update(measurement=[0.12, 0.08, 0.005])

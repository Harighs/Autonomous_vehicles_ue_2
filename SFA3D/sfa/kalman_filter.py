import numpy as np



class KalmanFilter3D:
    '''Kalman filter class for 3D tracking'''

    def __init__(self, dim_state=6, dim_meas=3):
        self.x = np.zeros((dim_state, 1))  # state vector [x, y, z, vx, vy, vz]
        self.P = np.eye(dim_state)  # initial uncertainty covariance
        self.Q = np.eye(dim_state)  # process noise covariance
        self.F = np.eye(dim_state)  # system dynamics matrix
        self.H = np.zeros((dim_meas, dim_state))  # measurement matrix
        self.R = np.eye(dim_meas)  # measurement uncertainty

        # Update system dynamics (F) for a constant velocity model
        dt = 1  # time step (e.g., 1 second)
        for i in range(3):
            self.F[i, i + 3] = dt

        # Update measurement matrix (H) to map state to position measurements
        for i in range(3):
            self.H[i, i] = 1

    def predict(self):
        self.x = self.F @ self.x  # state prediction
        self.P = self.F @ self.P @ self.F.T + self.Q  # covariance prediction
        return self.x, self.P

    def update(self, z):
        gamma = z - self.H @ self.x  # residual
        S = self.H @ self.P @ self.H.T + self.R  # residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ gamma  # update state estimate
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P  # update covariance
        return self.x, self.P

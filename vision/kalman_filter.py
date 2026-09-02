import numpy as np


class KalmanFilter2D:
    def __init__(self, dt=1.0 / 30.0, process_noise=1e-2, measurement_noise=1e-1):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1000.0

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.Q = np.array(
            [
                [0.25 * dt**4, 0.0, 0.5 * dt**3, 0.0],
                [0.0, 0.25 * dt**4, 0.0, 0.5 * dt**3],
                [0.5 * dt**3, 0.0, dt**2, 0.0],
                [0.0, 0.5 * dt**3, 0.0, dt**2],
            ],
            dtype=np.float64,
        ) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.initialized = False

    def initialize(self, x, y):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, x, y):
        if not self.initialized:
            self.initialize(x, y)
            return float(x), float(y)

        z = np.array([[x], [y]], dtype=np.float64)
        y_residual = z - np.dot(self.H, self.x)
        s = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        k = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(s))
        self.x = self.x + np.dot(k, y_residual)
        i = np.eye(4, dtype=np.float64)
        self.P = np.dot(i - np.dot(k, self.H), self.P)
        return float(self.x[0, 0]), float(self.x[1, 0])

    def reset(self):
        self.x[:] = 0.0
        self.P = np.eye(4, dtype=np.float64) * 1000.0
        self.initialized = False

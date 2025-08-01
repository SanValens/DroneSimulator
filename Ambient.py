import numpy as np

class Ambient:
    def __init__(self, dim = 2, speed = 0.0):
        self.dim = dim
        self.speed = speed

    def get_speed(self):
        """
        Returns the ambient wind speed as a numpy array.
        For simplicity, we assume a constant speed in the x direction (andincluding y-direction if 3D)
        and no wind in the z direction, for now
        """
        if self.dim == 2:
            return np.array([self.speed, 0.0])
        elif self.dim == 3:
            return np.array([self.speed, self.speed, 0.0])
        
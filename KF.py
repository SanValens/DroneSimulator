import filterpy
import numpy as np
from filterpy.kalman import KalmanFilter

'''
    Kalman filter programmed to estimate the state of the 3D drone.
    This version consists of a linear kalman filter with the following characterisitics

    Parameters:

    State: x =  [x, y, z, x_dot, y_dot, z_dot]
    Input: u = [x_dotdot, y_dotdot, z_dotdot] | Linear, fused and filtered acceleration vector from the accelerometer
    Possible measurements: z_1 = [x, y] or z_2 = [z], are the GPS and barometer readings respectively
 
    Given the current system, the equations that better describe the system are:

    x = Fx = Bu

    Where: F and B are the matrices in the __init__ functions

    About the other matrices:

    Q matrix = 
    
    

    Following modifications:
    *  
'''

class KF():
    def __init__(self, init_state, freq):

        dt = 1/ freq #Delta_t for prediction (the same as the computer processing speed)
        
        dim_x = 6
        dim_z = 2
        dim_u = 3
        self.x = init_state
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z, dim_u = dim_u)
        self.kf.x = self.x
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                              [0, 1, 0, 0, dt, 0],
                              [0, 0, 1, 0, 0, dt],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
        self.B = np.array([[(dt**2)/2, 0, 0],
                         [0, (dt**2)/2, 0],
                         [0, 0, (dt**2)/2],
                         [dt, 0, 0],
                         [0, dt, 0],
                         [0, 0, dt]])
        self.kf.F =  self.F
        
        self.kf.B = self.B
        
        self.kf.P = np.zeros((dim_x, dim_x)) #We are completaly certain about the initial state of the drone

        self.kf.Q = np.zeros((dim_x, dim_x)) 
        
        self.H_gps = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]])
        
        self.R_gps = np.zeros((2,2))
        
        self.H_baro = np.array([0, 0, 1, 0, 0, 0])

        self.R_baro = np.zeros(1)

    def step(self, u):
        #self.kf.predict(u = u)
        self.x = self.F @ self.x + self.B @ u
        return self.x
        #return self.kf.x[:3]
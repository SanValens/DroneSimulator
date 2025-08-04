from Drone import Drone2D
from Drone import motor_thurst
from Controller import Controller
import numpy as np
from Ambient import Ambient

class Simulation:
    def __init__(self, drone, ambient, dt=0.001):
        self.drone = drone
        self.ambient = ambient  
        self.dt = dt
        self.g = 9.81

        # Histories
        self.time_history = []
        self.state_history = []
        self.outputs = []
        self.error_history = []


    """
        Propagates the drone's dynamics in 2D space. When the dron is in the X condiguration.
        * Integrates the input of the motors anf of the ambient wind.
        * Computes the forces and torques acting on the drone.
    """
    def dynamics2D(self, state, u):
        # Unpack the state vector
        x, z, vx, vz, theta, omega = state
        
            # Translational dynamics
        # TODO: motor forces are computed as function of ESC signal, instead of using angular velocity. 
        motor_forces = motor_thurst(u)
        thrust_body = np.array([0, np.sum(motor_forces)])
        thrust_inertial = self.rotation_matrix2D(theta) @ thrust_body
        gravity_force = np.array([0, self.drone.m * self.g])

        # Calculate the ambient wind speed and its effect on the drone
        wind_speed = self.ambient.get_speed() # Returns a 2D array with wind speed in x and z directions
        wind_relative_speed =  np.array([vx, vz]) - wind_speed # drone speed relative to the wind
        ambient_force = -self.drone.c_d * np.linalg.norm(wind_relative_speed) * wind_relative_speed # drag force is opposite to the drone relative speed
        #print(f"Ambient force: {ambient_force}")
        
        # Merge forces
        forces_intertial = thrust_inertial - gravity_force + ambient_force

        
        ax = forces_intertial[0] / self.drone.m
        az = forces_intertial[1] / self.drone.m
        
            # Angular dynamics
        # TODO: Add the effect of the ambient wind on the angular dynamics
        torque = self.drone.l * (motor_forces[0] - motor_forces[1])
        alpha = torque / self.drone.I_yy

        return np.array([vx, vz, ax, az, omega, alpha])

    def run(self, t_max = 10):
        t_eval = np.arange(0, t_max, self.dt)

        for t in t_eval[1:]:
            # Ask the drone for control inputs
            u = self.drone.control(t)

            # Propagate the system using the current true state and control inputs
            new_state = step_RK4(self.dynamics2D, self.drone.state_true, u, self.dt)
            
            # Update the drone's true state
            self.drone.set_state_true(new_state)

        return t_eval
        

    def rotation_matrix2D(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
    
    def rotation_matrix3D(self, theta):
        pass


def step_RK4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


import numpy as np
from Controller import Controller
from Ambient import Ambient


"""
    * Drone class to represent a single dron. Includes its flight computers and control logic
    SOON: Include state estimation/ sensor reading.

    

    Attributes:
    - state_true: Real state of the drone, as a numpy array with the following order
        state = [x, z, vx, vz, theta, omega]
        [0]: position
        [1]: altitude
        [2]: velocity x
        [3]: velocity z
        [4]: angle
        [5]: angular velocity

    - state_hat: Estimated state of the drone, comes from the estimator, as a numpy array with the same order as state_true
    - estimator: State estimator algorithm instance (e.g., UKF)
    - controllers: Dictionary of PIDController instances for each property to control
    - m: Mass of the drone (kg)
    - I_yy: Moment of inertia around the y-axis (kg*m^2)
    - k: Lift constant of the motor-propeller system (N/(rad/s)^2)
    - l: Distance from the center of gravity to the motor (m)
    - c_d: Drag coefficient for translational motion
    - hover_speed: Speed of the motors at hover (m/s)
    - last_t: Last time step, used to calculate dt and control inputs. Used to simulate computer delays.
    - frequency: Frequency of the control loop (Hz). 10 Hz by default.
        
    - Methods:
        - control: Computes the control inputs based on the current state and PID controllers given.
           inputs: state (numpy array), dt (time step)
           returns: control inputs
        - readSensors: Simulates the raw reading of sensors.
           inputs: state 
"""

class Drone2D:
    def __init__(self, state_true, controllers, estimator, m=1.0, I_yy=0.1, k=0.1, l=0.2, c_d=0.3, frequency = 10, sync_mode = 'sync'):
        self.state_true = state_true
        self.state_hat = state_true.copy() # Initial estimate is true. Might change later.
        self.m = m
        self.I_yy = I_yy
        self.k = k
        self.l = l
        self.c_d = c_d
        self.controllers = controllers
        self.estimator = estimator
    

        #Time handling
        if(not (sync_mode == 'async' or  sync_mode == 'sync')):
            raise TypeError("Invalid sync_mode. Use 'async' or 'sync'.")
        else: self.sync_mode = sync_mode
        self.last_t = 0.0 # Last time step, used to calculate dt and
        self.frequency = frequency  # Control loop frequency in Hz
        self.computer_delay = 1.0 / frequency  # Time step for the control loop
        
        self.hover_speed = np.sqrt((m * 9.81 / 2) / k)
        self.hover_signals = np.array([self.hover_speed, self.hover_speed])  # Hover speed for both motors
        self.u = self.hover_signals 

        #Histories

        # Controllers errors history
        self.error_history = {
            'x_pos_errors': [0],
            'bank_errors': [0],
            'z_pos_errors': [0]
        }

        # Controller outputs history
        self.output_history = {
            'desired_bank': [0],
            'delta_u_x_pos': [0]
        }

        self.computer_time_history = [] # Times only for parameters that depend of computer processing
        self.state_true_history = []
        self.state_hat_history = []

        # Init histories
        self.computer_time_history.append(0.0)
        self.state_true_history.append(self.state_true)
        self.state_hat_history.append(self.state_hat)


    def control(self, t):
        """
        Computes control inputs using PID controllers in series.
        Handles both synchronous ('sync') and asynchronous ('async') operation modes.
        
        Args:
            t: Current simulation time
        
        Returns:
            Control inputs (motor commands)
        """
        # Calculate time since last control update
        sim_dt = t - self.last_t
        
        # Check if we should compute new control inputs. Either if we are running synchronized with the simulation
        # or if we are running asynchronously and the time since last control update is greater than the computer_delay. 
        if (self.sync_mode == 'sync') or (self.sync_mode == 'async' and sim_dt >= self.computer_delay):
            # Get current state estimate
            self.state_hat = self.estimateState()
            pos_x = self.state_hat[0]
            theta = self.state_hat[4]
            
            # Position controller (outputs desired angle)
            desired_bank, error_x_pos = self.controllers['pos_con'].PID(pos_x, sim_dt, 0)
            
            # Attitude controller (outputs motor differential)
            delta_u_x_pos, error_bank = self.controllers['bank_con'].PID(theta, sim_dt, -desired_bank)
            delta_u_series1 = np.array([delta_u_x_pos, -delta_u_x_pos])
            
            #Accumulate al delta_u given by in parallel controllers
            self.delta_u = delta_u_series1 #+
            
            # Apply hard limits to control inputs
            self.delta_u = np.clip(self.delta_u, -10000, 10000)

            # Update motor commands (hover + differential)
            self.u = self.hover_signals + self.delta_u
            
            self.last_t = t  # Update last control time

            # Store results for analysis
            self.computer_time_history.append(t)
            self.error_history['bank_errors'].append(error_bank)
            self.error_history['x_pos_errors'].append(error_x_pos)
            self.output_history['desired_bank'].append(desired_bank)
            self.output_history['delta_u_x_pos'].append(delta_u_x_pos)
        
        
        # Return current motor commands (either newly computed or previous)
        return self.u
            
    
    def readSensors(self):
        """
            Simulates the raw reading of sensors.
            Returns the current state of the drone.
        """
        return self.state_true
    
    def estimateState(self):
        """
            Updates the internal state of the drone with the given state.
            This is used to simulate the drone's state estimation.
        """
        z = self.readSensors()
        #self.state_hat = self.estimator.estimate(z, self.state_hat) When an estimator is implemented
        self.state_hat = self.state_true
        self.state_hat_history.append(self.state_hat)
        return self.state_hat

    def set_state_true(self, new_state_true):
        """
            Sets the true state of the drone.
            This is used to update the drone's state during the simulation.
        """
        self.state_true_history.append(new_state_true)
        self.state_true = new_state_true
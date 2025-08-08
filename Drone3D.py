import numpy as np
from Controller import Controller
from Ambient import Ambient

"""state : ndarray of shape (12,)
    The current state vector of the drone, in the following order:
    [0] x        - Position in x (inertial frame) [m]
    [1] y        - Position in y (inertial frame) [m]
    [2] z        - Position in z (inertial frame) [m]
    [3] x_dot    - Velocity in x (inertial frame) [m/s]
    [4] y_dot    - Velocity in y (inertial frame) [m/s]
    [5] z_dot    - Velocity in z (inertial frame) [m/s]
    [6] phi      - Roll angle [rad]
    [7] theta    - Pitch angle [rad]
    [8] psi      - Yaw angle [rad]
    [9] phi_dot  - Roll rate [rad/s]
    [10] theta_dot - Pitch rate [rad/s]
    [11] psi_dot - Yaw rate [rad/s]"""

class Drone3D:
    def __init__(self, state_true, controllers, estimator, flight_instructions, 
                 m=0.5, I = np.ones([3,3]), l=0.2, k_dt=np.zeros(3), k_dm = np.zeros(3), 
                 k_t = 0.02,k_m = 1e-5,k_f = 1.7e-2, J_r = 1e-4, frequency = 10, sync_mode = 'sync'):
        self.state_true = state_true
        self.state_hat = state_true.copy() # Initial estimate is true. Might change later.
        
        self.m = m
        self.I = I
        self.l = l
        self.k_dm = k_dm 
        self.k_dt = k_dt
        self.J_r = J_r 
        self.k_t = k_t  
        self.k_m = k_m 
        self.k_f = k_f


        self.controllers = controllers
        self.estimator = estimator
        self.I_xx = self.I[0, 0]
        self.I_yy = self.I[1, 1]
        self.I_zz = self.I[2, 2]


        #Time handling
        if(not (sync_mode == 'async' or  sync_mode == 'sync')):
            raise TypeError("Invalid sync_mode. Use 'async' or 'sync'.")
        else: self.sync_mode = sync_mode
        self.last_t = 0.0 # Last time step, used to calculate dt and
        self.frequency = frequency  # Control loop frequency in Hz
        self.computer_delay = 1.0 / frequency  # Time step for the control loop
        
        self.hov_sig = find_hover_signal(self.m * 9.81)
        print(f'The drone hovers when signal is: {self.hov_sig}')
        self.hover_input= np.array([self.hov_sig, self.hov_sig, self.hov_sig, self.hov_sig])  # Hover speed for both motors

        self.u = self.hover_input.copy()


        self.flight_instructions = flight_instructions
        #Histories

        # Controller outputs history
        self.input_history = {
            'delta_u_roll': [0],
            'delta_u_x_pos': [0],
            'delta_u_z_pos': [0]
        }

        self.computer_time_history = [0.0] # Times only for parameters that depend of computer processing
        self.state_true_history = [self.state_true]
        self.state_hat_history = [self.state_hat]
        self.motor_signal_history = {
            'motor1': [self.hov_sig],
            'motor2': [self.hov_sig],
            'motor3': [self.hov_sig],
            'motor4': [self.hov_sig] 
        }


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
            pos_y = self.state_hat[1]
            pos_z = self.state_hat[2]
            phi = self.state_hat[6]
            theta = self.state_hat[7]
            psi = self.state_hat[8]
            

            # Check if we have angular instructions, if we don't we just used whas is requeste by the position controllers
            if(self.flight_instructions['roll_instructions'] and self.flight_instructions['pitch_instructions']):
                desired_roll = self.flight_instructions['roll_instructions'](t)
                desired_pitch = self.flight_instructions['pitch_instructions'](t)
            else:
                # Pull the desired position in the Inertial frame
                desired_x = self.flight_instructions['x_instructions'](t)
                desired_y = self.flight_instructions['y_instructions'](t)
                desired_position_inertial = np.array([desired_x, desired_y])
                
                #Transform onto the body frame so we can map that position using roll and pitch commands
                desired_position_body = self.transformInertialToBodyPosition(desired_position_inertial, self.state_hat[8])

                #Operate the outerloop controllers
                desired_pitch = self.controllers['x_pos_con'].PID(pos_x, sim_dt, desired_position_body[0])
                desired_roll = self.controllers['y_pos_con'].PID(pos_y, sim_dt, desired_position_body[1])
                desired_pitch = desired_pitch * -1


            if(self.flight_instructions['yaw_instructions']):
                desired_yaw = self.flight_instructions['yaw_instructions'](t)
  
            delta_u_roll = self.controllers['roll_con'].PID(phi, sim_dt, desired_roll)
            delta_u_series1 = np.array([delta_u_roll, -delta_u_roll, -delta_u_roll, delta_u_roll])

            delta_u_pitch = self.controllers['pitch_con'].PID(theta, sim_dt, desired_pitch)
            delta_u_series2 = np.array([delta_u_pitch, delta_u_pitch, -delta_u_pitch, -delta_u_pitch])

            delta_u_yaw = self.controllers['yaw_con'].PID(psi, sim_dt, desired_yaw)
            delta_u_series3 = np.array([-delta_u_yaw, delta_u_yaw, -delta_u_yaw, delta_u_yaw])

            # Z position controller
            if(self.flight_instructions['z_instructions']):
                delta_u_z_pos = self.controllers['z_pos_con'].PID(pos_z, sim_dt, self.flight_instructions['z_instructions'](t))  # z-position control
                delta_u_series4 = np.array([delta_u_z_pos, delta_u_z_pos, delta_u_z_pos, delta_u_z_pos])
            else:
                delta_u_z_pos = 0
                delta_u_series4 = np.zeros(4)

            #Accumulate al delta_u given by in parallel controllers
            self.delta_u = delta_u_series1 +  delta_u_series2 + delta_u_series3 + delta_u_series4
            

            # Update motor commands (hover + differential)
            self.u = self.hover_input +  self.delta_u
            
            # Apply hard limits to control inputs
            self.u = np.clip(self.u, 0, 100)
            self.last_t = t  # Update last control time

            # Store results for analysis
            self.computer_time_history.append(t)
            self.input_history['delta_u_z_pos'].append(delta_u_z_pos)
            self.input_history['delta_u_roll'].append(delta_u_roll)
            self.motor_signal_history['motor1'].append(self.u[0])
            self.motor_signal_history['motor2'].append(self.u[1])
            self.motor_signal_history['motor3'].append(self.u[2])
            self.motor_signal_history['motor4'].append(self.u[3])
        
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

    def transformInertialToBodyPosition(self, pos ,yaw):
        c = np.cos
        s = np.sin
        yaw = yaw *-1
        R = np.array([
            [c(yaw), s(yaw)],
            [-s(yaw), c(yaw)]
        ])

        return R @ pos


def find_hover_signal(weight, tol=1e-10, max_iter=1000):
    # We want to find signal such that 4*thurst(signal) = weight
    # We'll use binary search between signal_min and signal_max
    
    signal_min = 0
    signal_max = 100  # You can adjust this max if needed
    
    for _ in range(max_iter):
        mid = (signal_min + signal_max) / 2
        thrust_total = 4 * motor_thrust3D(mid) # 4 for two motors
        
        if abs(thrust_total - weight) < tol:
            return mid
        elif thrust_total < weight:
            signal_min = mid
        else:
            signal_max = mid
    
    # If not converged, return the best guess
    return (signal_min + signal_max) / 2
        

def motor_thrust3D(signal):
    grams_force = 0.023*signal**2 + 1.945 * signal - 6.457
    grams_force = np.clip(grams_force, 0 , 0.023*100**2 + 1.945 * 100 - 6.457)
    newtons = 0.00981 * grams_force
    return newtons
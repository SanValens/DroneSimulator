import numpy as np
from Controller import Controller
from Ambient import Ambient
import random
from KF import KF


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
    def __init__(self, state_true, controllers, flight_instructions, sim_freq,
                 m=0.5, I = np.ones([3,3]), l=0.2, k_dt=np.zeros(3), k_dm = np.zeros(3), 
                 k_t = 0.02,k_m = 1e-5,k_f = 1.7e-2, J_r = 1e-4, frequency = 10, sync_mode = 'sync', gps_freq = 1, baro_freq = 20):
        
        # True shit

        self.i = 0
        self.state_true = state_true
        self.acc_true = state_true[6:9]

        self.vel_hat =  state_true[3:6]
        self.prev_vel_hat = self.vel_hat.copy()

        self.pos_hat = state_true[:3]
        self.euler_hat = np.array([0,0,0])
        
        # Estimated shit
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

        self.I_xx = self.I[0, 0]
        self.I_yy = self.I[1, 1]
        self.I_zz = self.I[2, 2]


        self.gps_delay = 1/gps_freq
        self.gps_time = 0
        self.baro_delay = 1/baro_freq
        self.baro_time = 0
        self.last_gps = 0
        self.last_baro = 0


        #Time handling
        if(not (sync_mode == 'async' or  sync_mode == 'sync')):
            raise TypeError("Invalid sync_mode. Use 'async' or 'sync'.")
        else: self.sync_mode = sync_mode


        self.last_t = 0.0 # Last time step, used to calculate dt and
        self.frequency = frequency  # Control loop frequency in Hz
        self.control_timer = 0.0

        if (self.sync_mode == 'sync'):
            print("Noticed that the settings are set to synchronous drone")
            print("Overwriting computer frequency in all aspects...")
            self.computer_delay = 1.0 / sim_freq
            print("New computer frequency: ", 1/self.computer_delay)
        else:
            self.computer_delay = 1.0 / frequency  # Time step for the control loop

        self.run_next = True


        #Kalman filter or estimator definition
        kf_state = self.state_true[:6]
        self.estimator = KF(kf_state, sim_freq)  # Initialize the KF instance

        #Control parameters definition
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
        
        self.acc_true_history = [self.acc_true]
        self.measurement_acc_history = [self.acc_true]
        self.state_true_history = [self.state_true]
        self.pos_hat_history = [self.pos_hat]
        self.euler_hat_history = [self.euler_hat]
        self.motor_signal_history = {
            'motor1': [self.hov_sig],
            'motor2': [self.hov_sig],
            'motor3': [self.hov_sig],
            'motor4': [self.hov_sig] 
        }


    def compute(self, t, sim_dt):
        # Calculate time since last control update
        if(self.control_timer + sim_dt >= self.computer_delay):
            required_dt = self.computer_delay - self.control_timer
            self.run_next = True
            self.control_timer = 0
            return required_dt
        else:
            self.control_timer += sim_dt
            return False
        
    def control(self, t):
        """
        Computes control inputs using PID controllers in series.
        Handles both synchronous ('sync') and asynchronous ('async') operation modes.
        
        Args:
            t: Current simulation time
        
        Returns:
            Control inputs (motor commands)
        """
        
        # Check if we should compute new control inputs. Either if we are running synchronized with the simulation
        # or if we are running asynchronously and the time since last control update is greater than the computer_delay. 
        if (self.run_next):
            self.run_next = False

            # Get current state estimate
            pos_hat, euler_hat = self.estimateState(t, self.computer_delay)

            pos_x = pos_hat[0]
            pos_y = pos_hat[1]
            pos_z = pos_hat[2]
            phi =   euler_hat[0]
            theta = euler_hat[1]
            psi =   euler_hat[2]
            

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
                desired_pitch = self.controllers['x_pos_con'].PID(pos_x, self.computer_delay, desired_position_body[0])
                desired_roll = self.controllers['y_pos_con'].PID(pos_y,  self.computer_delay , desired_position_body[1])
                desired_pitch = desired_pitch * -1


            if(self.flight_instructions['yaw_instructions']):
                desired_yaw = self.flight_instructions['yaw_instructions'](t)
  
            delta_u_roll = self.controllers['roll_con'].PID(phi,self.computer_delay, desired_roll)
            delta_u_series1 = np.array([delta_u_roll, -delta_u_roll, -delta_u_roll, delta_u_roll])

            delta_u_pitch = self.controllers['pitch_con'].PID(theta, self.computer_delay, desired_pitch)
            delta_u_series2 = np.array([delta_u_pitch, delta_u_pitch, -delta_u_pitch, -delta_u_pitch])

            delta_u_yaw = self.controllers['yaw_con'].PID(psi, self.computer_delay, desired_yaw)
            delta_u_series3 = np.array([-delta_u_yaw, delta_u_yaw, -delta_u_yaw, delta_u_yaw])

            # Z position controller
            if(self.flight_instructions['z_instructions']):
                delta_u_z_pos = self.controllers['z_pos_con'].PID(pos_z, self.computer_delay, self.flight_instructions['z_instructions'](t))  # z-position control
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
        
    def readPosition(self):

        """
            Simulates the reading of the GPS simplified in terms of position
        """
        pos = self.state_true[:3]
        return pos
    def readAccelerometerFusion(self):
        """
            Simulates the reading of the BNO055 accelerometer
            that has already passed through Sensor Fusion
        """
        acc_inertial = self.acc_true
        self.measurement_acc_history.append(acc_inertial)
        return acc_inertial
    
    def readEulerFusion(self):
        """
            Simulates the reading of the BNO055 euler angles
            given by the internal sensor fusion
        """
        noise = np.random.normal(0, 0.001, size=3)
        euler_angles = self.state_true[6:9] + noise
        return np.array(euler_angles)

    def readAltitude(self):
        """
            Simulates the reading of the MS5611 barometer
        """
        altitude_true = self.state_true[2]
        return np.array(altitude_true)
    
    def readGyroscope(self):
        """
            Simulates the reading of the BNO Gryroscope
        """
        euler_rate = self.state_true[3:6]
        return np.array(euler_rate)

    def estimateState(self, t, dt):
        """
            Updates the internal state of the drone with the given state.
            This is used to simulate the drone's state estimation.
        """

        acc_read_fusion = self.readAccelerometerFusion()

        self.pos_hat = self.estimator.step(acc_read_fusion)

        if(self.baro_time >= self.baro_delay):
            altitude_read =  self.readAltitude()
            self.baro_time = 0
        else:
            self.baro_time = self.baro_time + dt

        if(self.gps_time >= self.gps_delay):
            pos_read = self.readPosition()
            self.gps_time = 0
        else:
            self.gps_time = self.gps_time + dt

        
        self.euler_hat = self.readEulerFusion()
                
        
        self.pos_hat_history.append(self.pos_hat)
        self.euler_hat_history.append(self.euler_hat)

        self.state_hat = self.state_true
        
        return self.pos_hat, self.euler_hat

    def set_state_true(self, new_state_true):
        """
            Sets the true state of the drone.
            This is used to update the drone's state during the simulation.
        """
        self.state_true_history.append(new_state_true)
        self.state_true = new_state_true

    def set_output(self, output):
        """
            Sets the last output by the state space
            This is used to simulate a sensor and then estimate the state in the UKF
        """
        #Unpack the output to handle the variables easier
        acc_true = output

        self.acc_true_history.append(acc_true)
        self.acc_true = acc_true

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
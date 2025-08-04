import numpy as np
from Controller import Controller
from Ambient import Ambient

class Drone3D:
    def __init__(self, state_true, controllers, estimator, flight_instructions, 
                 m=0.5, I = np.ones([3,3]), l=0.2, k_dt=np.zeros(3), k_dm = np.zeros(3), 
                 k_t = 0.02,k_m = 1e-5,k_f = 1.7e-5, J_r = 1e-4, frequency = 10, sync_mode = 'sync'):
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
        self.I_xx, self.I_yy, self.I_zz = self.I





        #Time handling
        if(not (sync_mode == 'async' or  sync_mode == 'sync')):
            raise TypeError("Invalid sync_mode. Use 'async' or 'sync'.")
        else: self.sync_mode = sync_mode
        self.last_t = 0.0 # Last time step, used to calculate dt and
        self.frequency = frequency  # Control loop frequency in Hz
        self.computer_delay = 1.0 / frequency  # Time step for the control loop
        
        self.hov_sig = find_hover_signal(self.m * 9.81, self.k_t, self.k_f)
        print(f'The drone hovers when signal is: {self.hov_sig}')
        self.hover_input= np.array([self.hov_sig, self.hov_sig])  # Hover speed for both motors
        self.u = self.hover_input.copy()


        self.flight_instructions = flight_instructions
        #Histories

        # Controller outputs history
        self.input_history = {
            'desired_bank': [0],
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
            pos_z = self.state_hat[1]
            theta = self.state_hat[4]
            
            # X Position controller (outputs desired angle)
            if(self.flight_instructions['x_instruction']):
                desired_bank = self.controllers['x_pos_con'].PID(pos_x, sim_dt, self.flight_instructions['x_instruction'](t))
            else: 
                desired_bank = 0
            # Account for the fact that baking angle is measured counter-clockwise
            desired_bank = desired_bank * -1
            #desired_bank = np.clip(desired_bank, -1, 1) #Clip de desired angle

            # Attitude controller, innerloop (outputs motor differential)
            if(self.flight_instructions['bank_instruction']):
                desired_bank = self.flight_instructions['bank_instruction'](t)
            
            delta_u_x_pos = self.controllers['bank_con'].PID(theta, sim_dt, desired_bank)
            delta_u_series1 = np.array([delta_u_x_pos, -delta_u_x_pos])

            # Z position controller
            if(self.flight_instructions['z_instruction']):
                delta_u_z_pos = self.controllers['z_pos_con'].PID(pos_z, sim_dt, self.flight_instructions['z_instruction'](t))  # z-position control
                delta_u_series2 = np.array([delta_u_z_pos, delta_u_z_pos])
            else:
                delta_u_z_pos = 0
                delta_u_series2 = np.zeros(2)

            #Accumulate al delta_u given by in parallel controllers
            self.delta_u = delta_u_series1 + delta_u_series2
            

            # Update motor commands (hover + differential)
            self.u = self.hover_input + delta_u_series1 + delta_u_series2
            
            # Apply hard limits to control inputs
            self.u = np.clip(self.u, 0, 100)
            self.last_t = t  # Update last control time

            # Store results for analysis
            self.computer_time_history.append(t)
            self.input_history['desired_bank'].append(desired_bank)
            self.input_history['delta_u_x_pos'].append(delta_u_x_pos)
            self.input_history['delta_u_z_pos'].append(delta_u_z_pos)
            self.motor_signal_history['motor1'].append(self.u[0])
            self.motor_signal_history['motor2'].append(self.u[1])
        
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

# Find hover signal
def find_hover_signal(W, k_t, k_f):
    thrust_total = W
    thrust_per_motor = thrust_total / 4
    # Solve for omega: T = k_f * omega^2 => omega = sqrt(T / k_f)
    omega_hover = np.sqrt(thrust_per_motor / k_f)
    # Now invert the PWM-to-omega conversion (assuming linear: omega = k_t * u)
    u_hover = omega_hover / k_t
    return u_hover
        
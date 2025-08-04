from Drone import Drone2D
from Drone import motor_thurst
from Controller import Controller
import numpy as np
from Ambient import Ambient

class Simulation:
    def __init__(self, drone, ambient, dt=0.01):
        self.drone = drone
        self.ambient = ambient  
        self.dt = dt
        self.g = 9.81

        # Histories
        self.time_history = []
        self.state_history = []
        self.outputs = []
        self.error_history = []
        self.states = []


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
    
        #Drone configuration:
    """
        Front (x axis)
            ^
            |
    1 (CW)    2 (CCW)
            +          -> Y axis
    4 (CCW)   3 (CW)
            |

    Z axis is upwards
    """


    """
        Computes the dynamics:
        state: 12 element Numpy array with the following order:


        u: PWM signal input for each motor. Numpy array
    """
    def dynamics3D(self, state, u):
        """
        Computes the full 3D nonlinear dynamics of a quadrotor drone given the current state and motor PWM inputs.

        Parameters
        ----------
        state : ndarray of shape (12,)
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
            [11] psi_dot - Yaw rate [rad/s]

        u : ndarray of shape (4,)
            The PWM input signals to each of the four motors, in the following order:
            [0] Motor 1
            [1] Motor 2
            [2] Motor 3
            [3] Motor 4

        Returns
        -------
        state_dot : ndarray of shape (12,)
            The time derivative of the state vector, in the same order:
            [0] x_dot, [1] y_dot, [2] z_dot,
            [3] x_dotdot, [4] y_dotdot, [5] z_dotdot,
            [6] phi_dot, [7] theta_dot, [8] psi_dot,
            [9] phi_dotdot, [10] theta_dotdot, [11] psi_dotdot

        Notes
        -----
        - The function models both translational and rotational dynamics using Newton-Euler equations.
        - Includes gyroscopic effects from the rotating motors and aerodynamic drag forces/moments.
        - Assumes a 'X' quadrotor configuration with symmetric mass distribution and identical motors.
        """
        # 1. Unpack the state
        x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot = state

        # Repack
        # Velocity
        v = np.array([x_dot, y_dot, z_dot])
        # Euler angles and rates
        eta = np.array([phi, theta, psi])
        eta_dot = np.array([phi_dot, theta_dot, psi_dot])

        # Body frame rotation rates
        omega = self.euler_rate_transformation_matrix(eta, inverse = False) @ eta_dot
        # Unpack angular rates
        p, q, r = omega

        # 2. Estimate the angular velocity of each motor
        #   Taking into account the PWM - OMEGA relationship. 
        OMEGA_motor = self.drone.k_t * u
        #   Calculate the relative angular velocity of the motors to estimate the gyroscopic torques
        OMEGA_r =  -OMEGA_motor[0] + OMEGA_motor[1] - OMEGA_motor[2] + OMEGA_motor[3]

        # 3. Calculate thrurst for each motor 
        thrust_motors = self.drone.k_f * (OMEGA_motor)**2
        #   Unpack thurst
        T_1, T_2, T_3, T_4 = thrust_motors

        # 4. Start the calculations for the rotational dynamics
        # BLDC moments
        moment_motors = self.drone.k_m * (OMEGA_motor)**2

        # Unpack moments

        M_1, M_2, M_3, M_4 = moment_motors
        
        # Drone's body moments
        M_b = np.array([
            (T_1+T_4-T_2-T_3)*self.drone.l/np.sqrt(2),
            (T_1+T_2-T_3-T_4)*self.drone.l/np.sqrt(2),
            M_4 + M_2 - M_1 - M_3
        ])

        # Drone's drag moments
        M_a = self.drone.k_dm * eta_dot

        # ROTATIONAL DYNAMICS:

        
        I_omega = self.drone.I @ omega
        M_gyro = np.cross(omega, np.array([0, 0,self.drone.J_r * OMEGA_r]))
        omega_cross_Iomega = np.cross(omega, I_omega)
        omega_dot = np.linalg.inv(self.drone.I) @ (M_b - M_a - M_gyro - omega_cross_Iomega)



        """p_dot = b_1*(M_b[0] - M_a[0]) + (a_1 * r * q) - (q*a_2*OMEGA_r)
        q_dot = b_2*(M_b[1] - M_a[1]) + (a_3*p*r) + (p*a_4*OMEGA_r)
        r_dot = b_3*(M_b[2] - M_a[2]) + (a_5 * p * q)

        # Repack the omega
        omega_dot = [p_dot, q_dot, r_dot] """

        # Transform into inertial frame
        eta_dotdot = self.euler_rate_transformation_matrix(eta, inverse = True) @ omega_dot

        phi_dotdot, theta_dotdot, psi_dotdot = eta_dotdot

        # 5. Prepare for traslational dynamics
        # Traslational drag force
        F_a = self.drone.k_dt * np.linalg.norm(v) * v
        #Gravity effect
        gravity_force = np.array([0, 0, self.drone.m*self.g])
        
        # TRASLATIONAL DYNAMICS:
        T_motors = np.sum(thrust_motors)  # scalar total thrust
        thrust_vector_body = np.array([0, 0, T_motors])  # thrust in body frame (z-direction)
        a = (self.rotation_matrix(eta) @ thrust_vector_body - F_a - gravity_force)/self.drone.m

        x_dotdot, y_dotdot, z_dotdot = a

        # Final repackaging of the values

        state_dot = np.array([x_dot, y_dot, z_dot, x_dotdot, y_dotdot, z_dotdot, phi_dot, theta_dot, psi_dot, phi_dotdot, theta_dotdot, psi_dotdot])
        
        return state_dot
        

    def euler_rate_transformation_matrix(self, eta, inverse = False):
        """
        Returns the transformation matrix that maps Euler angle rates
        [phi_dot, theta_dot, psi_dot] to body angular rates [p, q, r].

        Parameters:
            phi:   Roll angle (in radians)
            theta: Pitch angle (in radians)

        Returns:
            3x3 numpy array: Transformation matrix T(phi, theta)
        """

        phi, theta, _ = eta
        c = np.cos
        s = np.sin

        T = np.array([
            [1, 0, -s(theta)],
            [0, c(phi), s(phi) * c(theta)],
            [0, -s(phi), c(phi) * c(theta)]
        ])

        if(inverse):
            return np.linalg.inv(T)
        else: 
            return T


    def rotation_matrix(self, eta):
        phi, theta, psi = eta
        c, s = np.cos, np.sin

        R = np.array([
            [c(theta)*c(psi),                      c(theta)*s(psi),                     -s(theta)],
            [s(phi)*s(theta)*c(psi) - c(phi)*s(psi),  s(phi)*s(theta)*s(psi) + c(phi)*c(psi),  s(phi)*c(theta)],
            [c(phi)*s(theta)*c(psi) + s(phi)*s(psi),  c(phi)*s(theta)*s(psi) - s(phi)*c(psi),  c(phi)*c(theta)]
        ])
        return R

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
    

    def run3D(self, t_max = 10):
        t_eval = np.arange(0, t_max, self.dt)
        states = np.zeros((int(t_max/self.dt), 12))
        i = 0
        # Simulation loop
        for t in t_eval:
            # Control input: hover for 5s, then increase thrust slightly
            if t < 5:
                u = np.ones(4) * (self.drone.hov_sig) + np.array([10, 10, 0, 0])
            else:
                u = np.ones(4)* (self.drone.hov_sig) + np.array([0,0,0,0]) 

            # Step dynamics
            new_state = step_RK4(self.dynamics3D, self.drone.state_true, u, self.dt)
            states[i, :] = self.drone.state_true = new_state
            i=i+1
        return t_eval, states
        
    

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


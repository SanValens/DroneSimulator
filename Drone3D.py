import numpy as np
import matplotlib.pyplot as plt

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
def dynamics3D(state, u):
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
    omega = euler_rate_transformation_matrix(eta, inverse = False) @ eta_dot
    # Unpack angular rates
    p, q, r = omega

    # 2. Estimate the angular velocity of each motor
    #   Taking into account the PWM - OMEGA relationship. 
    OMEGA_motor = k_t * u
    #   Calculate the relative angular velocity of the motors to estimate the gyroscopic torques
    OMEGA_r =  -OMEGA_motor[0] + OMEGA_motor[1] - OMEGA_motor[2] + OMEGA_motor[3]

    # 3. Calculate thrurst for each motor 
    thrust_motors = k_f * (OMEGA_motor)**2
    #   Unpack thurst
    T_1, T_2, T_3, T_4 = thrust_motors

    # 4. Start the calculations for the rotational dynamics
    # BLDC moments
    moment_motors = k_m * (OMEGA_motor)**2

    # Unpack moments

    M_1, M_2, M_3, M_4 = moment_motors
    
    # Drone's body moments
    M_b = np.array([
        (T_1+T_4-T_2-T_3)*l/np.sqrt(2),
        (T_1+T_2-T_3-T_4)*l/np.sqrt(2),
        M_4 + M_2 - M_1 - M_3
    ])

    # Drone's drag moments
    M_a = k_dm * eta_dot

    # ROTATIONAL DYNAMICS:

    
    I_omega = I @ omega
    M_gyro = np.cross(omega, np.array([0, 0, J_r * OMEGA_r]))
    omega_cross_Iomega = np.cross(omega, I_omega)
    omega_dot = np.linalg.inv(I) @ (M_b - M_a - M_gyro - omega_cross_Iomega)



    """p_dot = b_1*(M_b[0] - M_a[0]) + (a_1 * r * q) - (q*a_2*OMEGA_r)
    q_dot = b_2*(M_b[1] - M_a[1]) + (a_3*p*r) + (p*a_4*OMEGA_r)
    r_dot = b_3*(M_b[2] - M_a[2]) + (a_5 * p * q)

    # Repack the omega
    omega_dot = [p_dot, q_dot, r_dot] """

    # Transform into inertial frame
    eta_dotdot = euler_rate_transformation_matrix(eta, inverse = True) @ omega_dot

    phi_dotdot, theta_dotdot, psi_dotdot = eta_dotdot

    # 5. Prepare for traslational dynamics
    # Traslational drag force
    F_a = k_dt * np.linalg.norm(v) * v
    #Gravity effect
    gravity_force = np.array([0, 0, m*g])
    
    # TRASLATIONAL DYNAMICS:
    T_motors = np.sum(thrust_motors)  # scalar total thrust
    thrust_vector_body = np.array([0, 0, T_motors])  # thrust in body frame (z-direction)
    a = (rotation_matrix(eta) @ thrust_vector_body - F_a - gravity_force)/m

    x_dotdot, y_dotdot, z_dotdot = a

    # Final repackaging of the values

    state_dot = np.array([x_dot, y_dot, z_dot, x_dotdot, y_dotdot, z_dotdot, phi_dot, theta_dot, psi_dot, phi_dotdot, theta_dotdot, psi_dotdot])
    
    return state_dot
    

def euler_rate_transformation_matrix(eta, inverse = False):
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


def rotation_matrix(eta):
    phi, theta, psi = eta
    c, s = np.cos, np.sin

    R = np.array([
        [c(theta)*c(psi),                      c(theta)*s(psi),                     -s(theta)],
        [s(phi)*s(theta)*c(psi) - c(phi)*s(psi),  s(phi)*s(theta)*s(psi) + c(phi)*c(psi),  s(phi)*c(theta)],
        [c(phi)*s(theta)*c(psi) + s(phi)*s(psi),  c(phi)*s(theta)*s(psi) - s(phi)*c(psi),  c(phi)*c(theta)]
    ])
    return R


def step_RK4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)



if __name__ == '__main__':

    #Generales
    g = 9.81

    # - Características de la estructura
    # Físicas
    m = 0.5 # kg
    l = 11e-2 # 11cm of arms length
    I_zz = 0.03
    I_xx = I_yy = 5e-3 # Inertia in kg*m^2 #LACKS MEASUREMENT
    I_zz = 6e-3 # Inertia in kg*m^2 #LACKS MEASUREMENT
    I = np.diag([I_xx, I_yy, I_zz])
    # Aerodinámicas
    #k_dm = np.array([0.1, 0.1, 0.05]) # Moment drag coefficient vector (X, Y, Z)
    #k_dt = np.array([0.3, 0.3, 4]) # Traslation drag coefficient vector (X, Y, Z)
    k_dm = np.zeros(3) # Moment drag coefficient vector (X, Y, Z)
    k_dt = np.zeros(3) # Traslation drag coefficient vector (X, Y, Z)

    # - Características de los motores
    # Físicas
    J_r = 1e-4 # Rotor inertia #LACKS MEASUREMENT
    # Mecánicas
    k_t = 0.02 # PWM to rotor angular velocity constant #LACKS MEASUREMENT
    k_m = 1e-5 # Motors moment coefficient #LACKS MEASUREMENT
    k_f = 1.7e-5 # Motors thrust coefficient #LACKS CALIBRATION


    # We define some useful constants

    a_1 = (I_yy - I_zz) / I_xx
    a_2 = J_r / I_xx
    a_3 = (I_zz - I_xx) / I_yy
    a_4 = J_r / I_yy
    a_5 = (I_xx - I_yy) / I_zz
    b_1 = 1/I_xx
    b_2 = 1/I_yy
    b_3 = 1/I_zz



    # Simulation parameters
    dt = 0.01
    t_final = 10.0
    timesteps = int(t_final / dt)

    # Initial state: stationary, level
    state = np.zeros(12)

    # Find hover signal
    def find_hover_signal(m):
        thrust_total = m * g
        thrust_per_motor = thrust_total / 4
        # Solve for omega: T = k_f * omega^2 => omega = sqrt(T / k_f)
        omega_hover = np.sqrt(thrust_per_motor / k_f)
        # Now invert the PWM-to-omega conversion (assuming linear: omega = k_t * u)
        u_hover = omega_hover / k_t
        return u_hover

    u_hover = find_hover_signal(m)

    # Prepare storage
    states = np.zeros((timesteps, 12))
    times = np.linspace(0, t_final, timesteps)

    # Simulation loop
    for i in range(timesteps):
        t = i * dt

        # Control input: hover for 5s, then increase thrust slightly
        if t < 5:
            u = np.ones(4) * (u_hover) + np.array([10, 10, 0, 0])
        else:
            u = np.ones(4)* (u_hover) + np.array([0,0,0,0]) 

        # Step dynamics
        state = step_RK4(dynamics3D, state, u, dt)
        states[i, :] = state

    # Extract data
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    phi = states[:, 6]
    theta = states[:, 7]
    psi = states[:, 8]

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(times, x, label='x', alpha = 1 )
    plt.plot(times, y, label='y', alpha = 0.6)
    plt.plot(times, z, label='z', alpha = 0.6)
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(times, np.rad2deg(phi), label='Roll (φ)', alpha = 1)
    plt.plot(times, np.rad2deg(theta), label='Pitch (θ)', alpha = 0.6)
    plt.plot(times, np.rad2deg(psi), label='Yaw (ψ)', alpha = 0.6)
    plt.ylabel('Angle (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
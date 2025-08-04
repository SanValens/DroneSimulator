import numpy as np
import matplotlib.pyplot as plt

from flight_instructions import x_instructions, z_instructions
from UKF import UKF
from Controller import Controller
from Ambient import Ambient
from Drone3D import Drone3D
from Simulation import Simulation

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



    ukf = UKF()  # Initialize the UKF instance
    # Initial state: stationary, level
    state = np.zeros(12)

    controllers_dict = {
        
    }

    flight_instructions = {
        'x_instruction': x_instructions,
        'z_instruction': z_instructions,
        'bank_instruction': None
    }

    # Create a drone instance
    drone = Drone3D(state, [], ukf, flight_instructions, 
                 m, I, l, k_dt, k_dm, 
                 k_t,k_m,k_f, J_r, frequency = 10, sync_mode = 'sync')


    # Create an ambient instance (if needed)
    ambient = Ambient(dim = 2, speed = 0.0)
    
    #Define simulation parameters
    dt = 0.01  # Time step for the simulation
    t_max = 10  # Maximum simulation time in seconds
    
    # Create a simulation instance
    sim = Simulation(drone, ambient, dt)

    sim_time, states = sim.run3D(t_max)  # Run the simulation

    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    phi = states[:, 6]
    theta = states[:, 7]
    psi = states[:, 8]

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(sim_time, x, label='x', alpha = 1 )
    plt.plot(sim_time, y, label='y', alpha = 0.6)
    plt.plot(sim_time, z, label='z', alpha = 0.6)
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(sim_time, np.rad2deg(phi), label='Roll (φ)', alpha = 1)
    plt.plot(sim_time, np.rad2deg(theta), label='Pitch (θ)', alpha = 0.6)
    plt.plot(sim_time, np.rad2deg(psi), label='Yaw (ψ)', alpha = 0.6)
    plt.ylabel('Angle (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
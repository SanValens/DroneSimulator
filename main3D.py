import numpy as np
import matplotlib.pyplot as plt

from flight_instructions import x_instructions, y_instructions, z_instructions, roll_instructions, pitch_instructions, yaw_instructions

from KF import KF
from Controller import Controller
from Ambient import Ambient
from Drone3D import Drone3D
from Simulation import Simulation
from Anim3D import DroneVisualizer3D
from tools import double_integrate_to_function


if __name__ == '__main__':

    #Generales
    g = 9.81

    # - Características de la estructura
    # Físicas
    m = 0.5 # kg
    l = 11e-2 # 11cm of arms length
    I_xx = I_yy = 7e-2 # Inertia in kg*m^2 #LACKS MEASUREMENT
    I_zz = 6e-3 # Inertia in kg*m^2 #LACKS MEASUREMENT
    I = np.diag([I_xx, I_yy, I_zz])
    # Aerodinámicas
    k_dm = np.array([0.3, 0.3, 0.05]) # Moment drag coefficient vector (X, Y, Z)
    k_dt = np.array([0.3, 0.3, 0.5]) # Traslation drag coefficient vector (X, Y, Z)
    #k_dm = np.zeros(3) # Moment drag coefficient vector (X, Y, Z)
    #k_dt = np.zeros(3) # Traslation drag coefficient vector (X, Y, Z)

    # - Características de los motores
    # Físicas
    J_r = 1e-4 # Rotor inertia #LACKS MEASUREMENT
    # Mecánicas
    k_t = 0.02 # PWM to rotor angular velocity constant #LACKS MEASUREMENT
    k_m = 1e-2 # Motors moment coefficient #LACKS MEASUREMENT
    k_f = 1.7e-2 # Motors thrust coefficient #LACKS CALIBRATION

    # Initial state: stationary, level
    state = np.array([2,-3,1,    # POS
                      0.4,-0.1,0,    # VEL LINEAR
                      0,0,0,    # EULER ANGLES
                      0,0,0])   # EULER ANGLE RATE

    pid_gains_z_position = [8.5,1.2,3.4]
    pid_gains_yaw =  [0.2, 0.01, .1]

    pid_gains_x_position = [.1, 0., 0.2]
    pid_gains_pitch =  [4, 0., 7]
    
    pid_gains_y_position = [.1, 0., 0.2]
    pid_gains_roll =  [4, 0., 7]

    controllers_dict = {
        'yaw_con': Controller(pid_gains_yaw,          enabled = False),  #  controller
        'roll_con': Controller(pid_gains_roll,        enabled = False),  #  controller
        'pitch_con': Controller(pid_gains_pitch,      enabled = False),  #  controller
        'x_pos_con': Controller(pid_gains_x_position, enabled = False),  # x-position controller
        'y_pos_con': Controller(pid_gains_y_position, enabled = False),  # y-position controller
        'z_pos_con': Controller(pid_gains_z_position, enabled = False)  # z-position controller
    }

    flight_instructions = {
        'x_instructions': x_instructions,
        'y_instructions': y_instructions,
        'z_instructions': z_instructions,
        'roll_instructions': None,
        'pitch_instructions': None,
        'yaw_instructions': yaw_instructions
    }

    #Initialize Kalman filter
    computer_frequency = 30

    #Define simulation parameters
    sim_freq = 100
    dt = 1 / sim_freq   # Time step for the simulation
    t_max = 20  # Maximum simulation time in seconds

    # Create a drone instance
    drone = Drone3D(state, controllers_dict, flight_instructions, sim_freq,
                 m, I, l, k_dt, k_dm, 
                 k_t,k_m,k_f, J_r, frequency = computer_frequency, sync_mode = 'sync', gps_freq = 1, baro_freq = 20)


    # Create an ambient instance (if needed)
    ambient = Ambient(dim = 3, speed = 0.5)
    
    
    # Create a simulation instance
    sim = Simulation(drone, ambient, dt)

    sim_time = sim.run3D(t_max)  # Run the simulation

    # Extract results from the simulation and drone instances
    drone_computer_time = drone.computer_time_history

    drone_state_true = drone.state_true_history
    drone_acc_true = drone.acc_true_history
    drone_acc_measurement = drone.measurement_acc_history
    drone_pos_hat = drone.pos_hat_history
    drone_euler_hat = drone.euler_hat_history

    print('Amount of data in the sim time data: ', len(sim_time))
    print('Amount of data in the drone time data: ', len(drone_computer_time))

    z_pos_errors = controllers_dict['z_pos_con'].error_history
    
    z_pos_measurements = controllers_dict['z_pos_con'].measurement_history

    delta_u_z_pos = drone.input_history['delta_u_z_pos']
    delta_u_roll = drone.input_history['delta_u_roll']

    motor1_history = drone.motor_signal_history['motor1']
    motor2_history = drone.motor_signal_history['motor2']
    motor3_history = drone.motor_signal_history['motor3']
    motor4_history = drone.motor_signal_history['motor4']

    x_positions_true = [state[0] for state in drone_state_true]
    y_positions_true = [state[1] for state in drone_state_true]
    z_positions_true = [state[2] for state in drone_state_true]
    x_positions_hat = [state[0] for state in drone_pos_hat]
    y_positions_hat = [state[1] for state in drone_pos_hat]
    z_positions_hat = [state[2] for state in drone_pos_hat]
    roll_hat = [state[0] for state in drone_euler_hat]
    pitch_hat = [state[1] for state in drone_euler_hat]
    yaw_hat = [state[2] for state in drone_euler_hat]
    roll_true = [state[6] for state in drone_state_true] 
    pitch_true = [state[7] for state in drone_state_true] 
    yaw_true = [state[8] for state in drone_state_true] 

    x_acc_true = [state[0] for state in drone_acc_true]
    y_acc_true = [state[1] for state in drone_acc_true]
    z_acc_true = [state[2] for state in drone_acc_true]
 
    x_acc_measurement = [state[0] for state in drone_acc_measurement]
    y_acc_measurement = [state[1] for state in drone_acc_measurement]
    z_acc_measurement = [state[2] for state in drone_acc_measurement]



    motor_thrust_history = [motor1_history, motor2_history, motor3_history, motor4_history]  # Convert PWM to thrust


    #Integrate acceleration to see if we can get the same z position behavior that we got from the other side
    """ # Example usage:
    z_func = double_integrate_to_function(drone_computer_time, z_acc_true)
    z_estimated = z_func(drone_computer_time)
    """
    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(sim_time, x_positions_true, label='X position', alpha = 1 , linewidth = 2)
    plt.plot(sim_time, y_positions_true, label='Y position', alpha = 0.5 , linewidth = 2)
    plt.plot(sim_time, z_positions_true, label='Z position (altitude)', alpha = 0.8 )
    plt.plot(drone_computer_time, x_positions_hat, label='X position hat', linestyle = '-.', alpha = 0.5 )
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(sim_time, np.rad2deg(roll_true), label='Roll (φ)', alpha = 1)
    plt.plot(sim_time, np.rad2deg(pitch_true), label='Pitch (θ)', alpha = 0.3)
    plt.plot(sim_time, np.rad2deg(yaw_true), label='Yaw (ψ)', alpha = 0.3)
    #plt.plot(drone_computer_time, np.rad2deg(yaw_hat), label='Yaw estimated (ψ)', linestyle = '-.', alpha = 0.3)
    #plt.plot(drone_computer_time, np.rad2deg(roll_hat), label='Roll estimated (φ)', linestyle = '-.', alpha = 1)
    #plt.plot(drone_computer_time, np.rad2deg(pitch_hat), label='Pitch estimated (θ)', linestyle = '-.', alpha = 0.3)
    plt.ylabel('Angle (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    """ plt.plot(drone_computer_time, motor1_history, label='Signal for Motor 1 (Left forward)', color = 'red')
    plt.plot(drone_computer_time, motor2_history, label='Signal for Motor 2 (Right forward)', color = 'blue')
    plt.plot(drone_computer_time, motor3_history, label='Signal for Motor 1 (Right aft)', color = 'green')
    plt.plot(drone_computer_time, motor4_history, label='Signal for Motor 2 (Left aft)', color = 'pink')
    plt.axhline(y=100, color = 'k', linestyle='--')
    plt.axhline(y=drone.hover_input[0], color = 'gray', linestyle='--', label = 'Hover signal')
    plt.legend()
    plt.show() """

    plt.plot(sim_time, x_acc_true, label='Acceleration X', color = 'red', alpha = 0.3)
    plt.plot(drone_computer_time, x_acc_measurement, label='Acceleration X measured', color = 'purple', alpha = 0.3)
    #plt.plot(sim_time, y_acc_true, label='Acceleration Y', color = 'yellow')
    #plt.plot(sim_time, z_acc_true, label='Acceleration Z', color = 'blue')
    plt.legend()
    plt.show()

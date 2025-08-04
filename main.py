import numpy as np
import matplotlib.pyplot as plt

from flight_instructions import x_instructions, z_instructions
from UKF import UKF
from Controller import Controller
from Ambient import Ambient
from Drone import Drone2D
from Simulation import Simulation

if __name__ == "__main__":
    # Define drone parameters
    m = 0.5 #kg
    I_yy = 4.856 * 1e-3 #Inercia del dron completo
    l = 0.089 #m arm of the motors to the center of gravity
    c_d = 0.3 #Coeficiente de drag traslacional
    state = np.array([4, 0, 0, 0, -np.pi/5, 0])
    #state = np.array([0, 0, 0, 0, 0, 0])

    # Define PID gains for each controller
    #OLD PIDS THAT WORKED WITH NORMAL DERIVATIVE
    pid_gains_banking =  [20, 0., 5.26764852]
    pid_gains_x_position = [0.5, 0.4, 0.5]
    pid_gains_z_position = [13.5,10.3,5.4]

    # Create instances of controllers and group them in a dictionary
    controllers_dict = {
        'bank_con': Controller(pid_gains_banking, enabled = True),  # Banking controller
        'x_pos_con': Controller(pid_gains_x_position, enabled = True),  # x-position controller
        'z_pos_con': Controller(pid_gains_z_position, enabled = True)  # z-position controller
    }

    flight_instructions = {
        'x_instruction': x_instructions,
        'z_instruction': z_instructions,
        'bank_instruction': None
    }

    ukf = UKF()  # Initialize the UKF instance

    # Create a drone instance
    drone = Drone2D(state, controllers_dict, ukf, flight_instructions, m, I_yy, l, c_d, sync_mode = 'sync', frequency=20)
    
    # Create an ambient instance (if needed)
    ambient = Ambient(dim = 2, speed = 0.0)
    
    #Define simulation parameters
    dt = 0.005  # Time step for the simulation
    t_max = 40  # Maximum simulation time in seconds
    
    # Create a simulation instance
    sim = Simulation(drone, ambient, dt)

    sim_time = sim.run(t_max)  # Run the simulation

    # Extract results from the simulation and drone instances
    drone_computer_time = drone.computer_time_history

    drone_state_true = drone.state_true_history
    drone_state_hat = drone.state_hat_history

    bank_errors = controllers_dict['bank_con'].error_history
    x_pos_errors = controllers_dict['x_pos_con'].error_history
    z_pos_errors = controllers_dict['z_pos_con'].error_history
    
    bank_measurements = controllers_dict['bank_con'].measurement_history
    x_pos_measurements = controllers_dict['x_pos_con'].measurement_history
    z_pos_measurements = controllers_dict['z_pos_con'].measurement_history

    bank_desired =  drone.input_history['desired_bank']
    delta_u_x_pos = drone.input_history['delta_u_x_pos']
    delta_u_z_pos = drone.input_history['delta_u_z_pos']

    motor1_history = drone.motor_signal_history['motor1']
    motor2_history = drone.motor_signal_history['motor2']

    x_positions_true = [state[0] for state in drone_state_true]
    bank_true = [state[4] for state in drone_state_true]
    z_positions_true = [state[1] for state in drone_state_true]

    x_positions_hat = [state[0] for state in drone_state_hat]
    z_positions_hat = [state[1] for state in drone_state_hat]

    max_bank = max(np.abs(bank_true))

    if(max_bank > 1):
        print(f'Maximum bank angle over 60 deg ({max_bank*180/np.pi} deg). Unsuccessful simulation')
    else: print(f'Fell into limits. Success')    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    plt.plot(sim_time, x_positions_true, label='True x position', color='blue')
    plt.plot(sim_time, z_positions_true, label='True z position', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--')


    plt.subplot(4, 1, 2)
    plt.xlabel('Time (s)')
    plt.ylabel('Bank Angle (rad)')
    plt.plot(sim_time, bank_true, label='True Bank angle', color='green')
    plt.scatter(drone_computer_time, bank_desired, label='Desired Bank angle', color='orange', s=5)
    plt.title('Drone Bank angle Over Time')
    plt.axhline(y=0, color='k', linestyle='--')

    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(drone_computer_time, x_pos_errors, label='X Position Error', color='orange')
    plt.plot(drone_computer_time, z_pos_errors, label='Z Position Error', color='green')
    plt.plot(drone_computer_time, bank_errors, label='Bank Error', color='purple')
    plt.title('X Position Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m or rad)')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(drone_computer_time, delta_u_x_pos, label='Control input for X controller', color='orange')
    plt.plot(drone_computer_time, delta_u_z_pos, label='Control input for Z controller', color='green')

    plt.title('Delta signal for every series of controllers')
    plt.xlabel('Time (s)')
    plt.ylabel('%')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.plot(drone_computer_time, motor1_history, label='Signal for Motor 1 (right)', color = 'red')
    plt.plot(drone_computer_time, motor2_history, label='Signal for Motor 2 (left)', color = 'blue')
    plt.axhline(y=100, color = 'k', linestyle='--')
    plt.axhline(y=drone.hover_input[0], color = 'gray', linestyle='--', label = 'Hover signal')
    plt.legend()
    plt.show()



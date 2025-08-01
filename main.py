import numpy as np
import matplotlib.pyplot as plt

from UKF import UKF
from Controller import Controller
from Ambient import Ambient
from Drone import Drone2D
from Simulation import Simulation

if __name__ == "__main__":
    # Define drone parameters
    m = 0.468 #kg
    I_yy = 4.856 * 1e-3 #Inercia del dron completo
    k = 2.980 * 1e-6 #Constante de lift
    l = 0.225 #m arm of the motors to the center of gravity
    c_d = 0.0 #Coeficiente de drag traslacional
    #state = np.array([4, 0, 0, 0, -np.pi/5, 0])
    state = np.array([0, 0, 1, 0, 0, 0])

    # Define PID gains for each controller
    pid_gains_banking =  [0.5, 0., 4.5]
    pid_gains_position = [0.5, 0.4, 1.5]

    # Create instances of controllers and group them in a dictionary
    controllers_dict = {
        'bank_con': Controller(pid_gains_banking, enabled = True),  # Banking controller
        'pos_con': Controller(pid_gains_position, enabled = True)  # Position controller
    }

    ukf = UKF()  # Initialize the UKF instance

    # Create a drone instance
    drone = Drone2D(state, controllers_dict, ukf, m, I_yy, k, l, c_d, sync_mode = 'async', frequency=20)
    
    # Create an ambient instance (if needed)
    ambient = Ambient(dim = 2, speed = 0.0)
    
    #Define simulation parameters
    dt = 0.001  # Time step for the simulation
    t_max = 40  # Maximum simulation time in seconds
    
    # Create a simulation instance
    sim = Simulation(drone, ambient, dt)

    sim_time = sim.run(t_max)  # Run the simulation

    # Extract results from the simulation and drone instances
    drone_computer_time = drone.computer_time_history

    drone_state_true = drone.state_true_history
    drone_state_hat = drone.state_hat_history

    x_positions_true = [state[0] for state in drone_state_true]
    z_positions_true = [state[1] for state in drone_state_true]

    x_positions_hat = [state[0] for state in drone_state_hat]
    z_positions_hat = [state[1] for state in drone_state_hat]
    
    # Plotting the results
    plt.plot(sim_time, x_positions_true, label='True x position', color='blue')
    plt.scatter(drone_computer_time, x_positions_hat, label='Estimated x position', color='red', s = 4)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.legend()
    plt.show()
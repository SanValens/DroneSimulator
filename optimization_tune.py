import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from flight_instructions import x_instructions, z_instructions, bank_instructions
from UKF import UKF
from Controller import Controller
from Ambient import Ambient
from Drone import Drone2D
from Simulation import Simulation

def simulate_with_gains(gains):
    # Define drone parameters
    m = 0.5 #kg
    I_yy = 4.856 * 1e-3 #Inercia del dron completo
    l = 0.089 #m arm of the motors to the center of gravity
    c_d = 0.3 #Coeficiente de drag traslacional
    state = np.array([0, 0, 0, 0, np.pi/6, 0])

    # Define PID gains for each controller
    #OLD PIDS THAT WORKED WITH NORMAL DERIVATIVE
    pid_gains_banking =  gains
    pid_gains_x_position = [0.5, 0.4, 0.5]
    pid_gains_z_position = [13.5,10.3,5.4]
    print(gains)

    # Create instances of controllers and group them in a dictionary
    controllers_dict = {
        'bank_con': Controller(pid_gains_banking, enabled = True),  # Banking controller
        'x_pos_con': Controller(pid_gains_x_position, enabled = False),  # x-position controller
        'z_pos_con': Controller(pid_gains_z_position, enabled = False)  # z-position controller
    }

    flight_instructions = {
        'x_instruction': x_instructions,
        'z_instruction': z_instructions,
        'bank_instruction': bank_instructions # should be != from 0
    }
    ukf = UKF()  # Initialize the UKF instance

    # Create a drone instance
    drone = Drone2D(state, controllers_dict, ukf, flight_instructions, m, I_yy, l, c_d, sync_mode = 'sync', frequency=20)
    
    # Create an ambient instance (if needed)
    ambient = Ambient(dim = 2, speed = 0.0)
    
    #Define simulation parameters
    dt = 0.005  # Time step for the simulation
    t_max = 10  # Maximum simulation time in seconds
    
    # Create a simulation instance
    sim = Simulation(drone, ambient, dt)

    sim_time = sim.run(t_max)  # Run the simulation

    z_pos_errors = controllers_dict['z_pos_con'].error_history

    cost = np.sum(np.square(z_pos_errors))

    print(cost)

    return cost
    




if __name__ == "__main__":
    # Initial guess for PID gains: [Kp, Ki, Kd]
    initial_guess = [.5, 0.0, 3.0]

    # Bounds for PID gains to keep them within reasonable limits
    bounds = [(0.0, 20.0), (0.0, 10.0), (0.0, 10.0)]

    # Perform optimization
    result = minimize(
    simulate_with_gains,
    initial_guess,
    bounds=bounds,
    method='L-BFGS-B',
)

    if result.success:
        optimal_gains = result.x
        print("Optimal PID Gains for banking controller:", optimal_gains)
    else:
        print("Optimization failed:", result.message)
import numpy as np
import matplotlib.pyplot as plt




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
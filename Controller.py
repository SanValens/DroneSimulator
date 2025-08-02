import numpy as np

# PID Controller
"""
    Holds indiviual controllers parameteres. One for each flight property to control.

    INPUTS:
    - gains: List of PID gains [Kp, Ki, Kd]

    METHODS:
    - PID: Computes the control output based on the current measurement and setpoint.

    OUTPUTS:
    - control_output: The control output computed by the PID controller, the error used.
"""

class Controller:
    def __init__(self, gains, enabled=True):
        self.Kp, self.Ki, self.Kd = gains
        self.last_error = None
        self.last_measurement = None
        self.integral = 0
        self.enabled = enabled

    def PID(self, measurement, dt, setpoint):
        error = setpoint - measurement
        self.integral += error * dt

        #Derivative-kick sensitive code
        if self.last_error is None:
          derivative = 0
        else:
          derivative = (error - self.last_error) / dt
        
        self.last_error = error
        
        """ # Derivative of the measurement to avoid derivative kick
        if self.last_measurement is None:
            derivative = 0
        else:
            derivative = -(measurement - self.last_measurement) / dt  # Note the negative sign """

        self.last_measurement = measurement

        if not self.enabled:
            return 0, error
        else : return self.Kp*error + self.Ki*self.integral + self.Kd*derivative, error

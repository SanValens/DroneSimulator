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
        self.last_setpoint = 0
        self.setpoint_change_threshold = 0.1
        
        self.error_history = [0]
        self.measurement_history = [0]

    def PID(self, measurement, dt, setpoint):
        error = setpoint - measurement
        self.integral += error * dt

        #Derivative-kick sensitive code
            #Use relative error to tell if the setpoint was changed abruptly
        if (self.last_error) is None or (setpoint-self.last_setpoint)/(setpoint+0.0001) > self.setpoint_change_threshold: 
          derivative = 0
        else:
          derivative = (error - self.last_error) / dt
        
        
        
        self.last_error = error


        #To avoid derivati
        
        """ # Derivative of the measurement to avoid derivative kick
        if self.last_measurement is None:
            derivative = 0
        else:
            derivative = -(measurement - self.last_measurement) / dt  # Note the negative sign """

        self.last_measurement = measurement
        self.last_setpoint = setpoint
        self.measurement_history.append(measurement)
        self.error_history.append(error)

        if not self.enabled:
            return 0
        else : return self.Kp*error + self.Ki*self.integral + self.Kd*derivative

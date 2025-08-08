import sympy as sp
import numpy as np
# Define time variable
t = sp.symbols('t')

# Define Euler angles as functions of time
phi = sp.Function('phi')(t)
theta = sp.Function('theta')(t)

# Define the transformation matrix T(phi, theta)
T = sp.Matrix([
    [1, 0, -sp.sin(theta)],
    [0, sp.cos(phi), sp.sin(phi) * sp.cos(theta)],
    [0, -sp.sin(phi), sp.cos(phi) * sp.cos(theta)]
])

# Differentiate the matrix element-wise
T_inv = T.inv()

# Optionally, simplify
T_dot_simplified = sp.simplify(T_inv)

# Display result
sp.pprint(T_dot_simplified)

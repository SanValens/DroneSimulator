import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# Función recursiva para aplicar cuadratura adaptativa (basada en Simpson 1/3)
def adaptiva(f, a, b, fa, fb, fm, tol, t, thrust, profundidad=0, max_profundidad=10):
    h = b - a
    c = (a + b) / 2  # Punto medio del intervalo
    # Se evalúan puntos adicionales en los subintervalos para mayor precisión
    fd = np.interp((a + c)/2, t, thrust)
    fe = np.interp((c + b)/2, t, thrust)

    # Estimación con una sola aplicación de Simpson (I1)
    I1 = h/6 * (fa + 4*fm + fb)
    # Estimación dividiendo el intervalo en dos partes (I2)
    I2 = h/12 * (fa + 4*fd + 2*fm + 4*fe + fb)

    # Si la diferencia es menor a la tolerancia o se alcanza la profundidad máxima, se acepta la estimación
    if abs(I2 - I1) < 15 * tol or profundidad >= max_profundidad:
        return I2
    else:
        # Si no, se divide el intervalo y se aplica recursivamente en cada mitad
        return (adaptiva(f, a, c, fa, fm, fd, tol/2, t, thrust, profundidad+1) +
                adaptiva(f, c, b, fm, fb, fe, tol/2, t, thrust, profundidad+1))

# Función principal para ejecutar la cuadratura adaptativa
def cuadratura_adaptativa(t, thrust, tol):
    a, b = t[0], t[-1]  # Intervalo de integración
    fa = np.interp(a, t, thrust)
    fb = np.interp(b, t, thrust)
    fm = np.interp((a + b)/2, t, thrust)
    # Llamada inicial a la función recursiva
    I = adaptiva(None, a, b, fa, fb, fm, tol, t, thrust)
    return I



def double_integrate_to_function(t, acc, v0=0.0, z0=0.0):
    t = np.array(t)
    acc = np.array(acc)


    # First integration: acceleration -> velocity
    vel = v0 + cumulative_trapezoid(acc, t, initial=0)

    # Second integration: velocity -> position
    pos = z0 + cumulative_trapezoid(vel, t, initial=0)

    # Return interpolating function for position
    return interp1d(t, pos, kind='cubic', fill_value="extrapolate")

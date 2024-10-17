import numpy as np

SQRT_TWO_PI = np.sqrt(2 * np.pi)
# I KNOW this doesnt make sense, its from when I tried fitting something before 
def gaussian(x, A, B):
    y = np.exp(-0.5 * ((x-B)/A)**2) / (A * SQRT_TWO_PI)
    return y

def cubic(x, A, B, C, D):
    y = A + B * (x) + C * (x**2) + D * (x**3)
    return y

def quartic(x, A, B, C, D, E):
    y = A + B * (x) + C * (x**2) + D * (x**3) + E * (x**4)
    return y

def invexp(x, A, B):
    y = 1 / (1 + np.exp(- A * (x - B)))
    return y

def invexp2(x, A, B, C):
    y = 1 / (1 + C * np.exp(- A * (x - B)))
    return y
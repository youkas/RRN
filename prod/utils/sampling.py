import math

from smt import sampling_methods
import numpy as np


def LHS(span, dimension, size):
    return sampling_methods.LHS(xlimits=np.array([span for i in range(dimension)]))(size)

def PLHS(span, dimension, size):
    #https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    radius = np.random.uniform(span[0], span[1], (size, 1))
    angles = sampling_methods.LHS(xlimits=np.array([[0., 2*math.pi] for i in range(dimension - 1)]))(size)
    cos_angles = np.hstack([np.cos(angles), np.ones((size, 1))])
    sin_angles = np.hstack([np.ones((size, 1)), np.cumprod(np.sin(angles), axis=1)])
    return radius * sin_angles * cos_angles


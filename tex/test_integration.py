"""
Written by Simon Hu, all rights reserved.
"""

import time
import numpy as np
from shapely.geometry import *
import scipy.integrate as scint
from research_utils.vorutils import helpers
from research_utils.vorutils import calctools

def test_integration_basic():
    mu = [0.7]
    sigma = 0.2
    func = lambda x : (1/(2*np.pi*sigma**2)) * np.exp(-(x-mu[0])**2/(2*sigma**2))

    start = time.time()
    val = scint.fixed_quad(func, 0, 2, n=11)
    end = time.time()
    print(end - start)
    print(val)

    start = time.time()
    val = scint.quad(func, 0, 2)
    end = time.time()
    print(end - start)
    print(val)

    start = time.time()
    val, s = calctools.gl_quadrature(func, 0, 2, ord=11)
    end = time.time()
    print(end - start)
    print(val)

def test_gaussian_integral():
    mu = [0.7, 0.2]
    sigma = 1

    func = lambda y,x : (1/(2*np.pi*sigma**2)) * np.exp(-((x-mu[0])**2 + (y-mu[1])**2)/(2*sigma**2))
    valscint = scint.dblquad(func, -1, 1, lambda x : -1, lambda x : 1)
    print(valscint)

if __name__ == "__main__":
    test_gaussian_integral()

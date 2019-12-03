"""
Python file to create a 'simulation' of the Voronoi diagrams.
"""

""" This file is depreciated. """

__author__ = "Simon Hu"
__copyright__ = "Copyright (c) 2019, Simon Hu"
__maintainer__ = "Simon Hu"
__email__ = "simonhu@ieee.org"

import time
import test_utils
import numpy as np
from shapely.geometry import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from research_utils.vorutils import helpers
from scipy.stats import multivariate_normal

if __name__ == "__main__":

    # create 2 kernels
    m1 = (-1,-1)
    s1 = np.eye(2)
    k1 = multivariate_normal(mean=m1, cov=s1)

    m2 = (1,1)
    s2 = np.eye(2)
    k2 = multivariate_normal(mean=m2, cov=s2)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-3, 3)
    ylim = (-3, 3)
    xres = 1000
    yres = 1000

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy) + k2.pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres,yres))
    img = resize(img, (80, 80))
    plt.imshow(img, cmap='YlGnBu'); plt.show()

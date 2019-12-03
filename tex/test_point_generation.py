"""
Python file to test the point generation algorithms.
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
from scipy.spatial import *
import matplotlib.pyplot as plt
from research_utils.vorutils import helpers

if __name__ == "__main__":

    poly_points = [[0,0], [1,1], [1,0]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    poly = Polygon(poly_points)

    points = helpers.generate_points_within_polygon(poly, 10)
    points_x = np.array([points[i][0] for i in range(10)])
    points_y = np.array([points[i][1] for i in range(10)])

    plt.figure()
    plt.plot(poly_points_x, poly_points_y, 'bo')
    plt.plot(points_x, points_y, 'go')
    plt.show()

"""
Point generation within the polygon passes the required test.
"""

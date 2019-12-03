"""
Python file to create a simulation tracking the centroids over time.
"""

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

def track_centroids_naive(num_samples=2, num_frames=100):
    """
    Tracks the location of the centroids as the Voronoi diagram changes with respect to time.

    Parameters
    ----------
    num_samples : int, optional
        Number of agents to populate the environment with.
    num_frames : int, optional
        Number of frames for the simulation.
    """
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)

    points = helpers.generate_points_within_polygon(main_poly, num_points=num_samples)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    ext_x, ext_y = main_poly.exterior.xy
    plt.plot(ext_x, ext_y, color='black', linewidth=0.7)

    for k in range(num_frames):
        vorinfo, centroids = test_utils.find_new_voronoi_points_for_polygon(main_poly, points)
        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        plt.plot(centroids_x, centroids_y, 'bo', markersize=0.2)
        plt.axis('equal')

        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        if k == num_frames-1:
            # Draw the final Voronoi configuration
            for region in vorinfo[0]:
                polygon_points = vorinfo[1][region]
                poly = Polygon(polygon_points).intersection(main_poly)
                polygon = [p for p in poly.exterior.coords]
                plt.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.7)

        plt.show(block=False)
        plt.pause(0.1)

    plt.savefig('naive_simulation_2.png')

def track_centroids_gaussian(num_samples=10, num_frames=100):
    """
    Tracks the movement of the centroids when the density function is not
    constant; for our purposes, we will just consider a two dimensional
    Gaussian.
    """
    pass

if __name__ == "__main__":
    track_centroids_naive(num_samples=10)

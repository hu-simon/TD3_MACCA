"""
Written by Simon Hu, all rights reserved.
"""

import time
import test_utils
import numpy as np
from scipy.spatial import *
from shapely.geometry import *
import matplotlib.pyplot as plt
from research_utils.vorutils import helpers

def voronoi_movie(num_samples=100, num_frames=100):
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

    sample_bounds_1 = [[20,20], [40,20], [20,40], [40,40]]
    sample_bounds_2 = [[-20,-20], [-40,-20], [-20,-40], [-40,-40]]
    sample_poly_1 = Polygon(sample_bounds_1)
    sample_poly_2 = Polygon(sample_bounds_2)

    points1 = helpers.generate_points_within_polygon(sample_poly_1, num_points=num_samples)
    points2 = helpers.generate_points_within_polygon(sample_poly_2, num_points=num_samples)
    points = points1 + points2
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    ext_x, ext_y = main_poly.exterior.xy
    plt.plot(ext_x, ext_y, color='black', linewidth=0.7)

    for k in range(num_frames):
        vorinfo, centroids = test_utils.find_new_voronoi_points_for_polygon(main_poly, points)
        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        plt.plot(centroids_x, centroids_y, 'ro', markersize=0.1)
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

    plt.savefig('close_naive_simulation_1.png')

if __name__ == "__main__":
    voronoi_movie(num_samples=50)

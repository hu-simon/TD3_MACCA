"""
Python file testing helper functions from vorutils.
"""

__author__ = "Simon Hu"
__copyright__ = "Copyright (c) 2019, Simon Hu"
__maintainer__ = "Simon Hu"
__email__ = "simonhu@ieee.org"

import time
import numpy as np
from shapely.ops import *
from scipy.spatial import *
from shapely.geometry import *
import matplotlib.pyplot as plt
from research_utils.vorutils import helpers

def test_finite_voronoi(a, b, num_samples, stretch_x=10, stretch_y=10):
    """
    Tests whether the output of the helper.create_finite_voronoi_2d() function
    returns what we expect.
    """

    points = helpers.generate_random_points(a, b, num_samples)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    vor = Voronoi(points)
    regions, vertices = helpers.create_finite_voronoi_2d(vor)

    min_x = vor.min_bound[0] - stretch_x
    max_x = vor.max_bound[0] + stretch_x
    min_y = vor.min_bound[1] - stretch_y
    max_y = vor.max_bound[1] + stretch_y
    bounding_box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    # Find the centroids and plot them in the diagram.
    centroids = list()

    # Colorize the regions.
    for region in regions:
        polygon_points = vertices[region]
        poly = Polygon(polygon_points).intersection(bounding_box)
        centroids.append(poly.centroid.xy)
        polygon = [p for p in poly.exterior.coords]
        plt.fill(*zip(*polygon), alpha=0.8, facecolor='none', edgecolor='black', linewidth=1)

    centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
    centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

    plt.plot(points_x, points_y, 'ko', markersize=3)
    plt.plot(centroids_x, centroids_y, 'bo', markersize=3)
    plt.axis('equal')

    plt.show()

def test_finite_voronoi_with_polygon(num_samples):
    """
    Tests whether the output of the helper.create_finite_voronoi_2d() function
    returns what we expect, when we bound the domain using a convex polygon.
    """

    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)

    points = helpers.generate_points_within_polygon(main_poly, num_points=num_samples)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    vor = Voronoi(points)
    regions, vertices = helpers.create_finite_voronoi_2d(vor)

    centroids = list()

    for region in regions:
        polygon_points = vertices[region]
        poly = Polygon(polygon_points).intersection(main_poly)
        centroids.append(poly.centroid.xy)
        polygon = [p for p in poly.exterior.coords]
        plt.fill(*zip(*polygon), alpha=0.8, facecolor='none', edgecolor='black', linewidth=1)

    centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
    centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

    plt.plot(points_x, points_y, 'ko', markersize=3)
    plt.plot(centroids_x, centroids_y, 'bo', markersize=3)
    plt.axis('equal')

    plt.show()

if __name__ == "__main__":

    test_finite_voronoi(0, 20, 10, stretch_x=10, stretch_y=10)
    test_finite_voronoi_with_polygon(num_samples=15)

"""
Both tests pass.
"""

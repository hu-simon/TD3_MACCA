"""
Python file to create a 'simulation' of the Voronoi diagrams.
"""

"""
Need to start thinkng about integrating cost functions and other information into the program.
Though it is nice that the diagram works for some general convex polygon. 
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

def test_with_bounding_box(a, b, num_samples, stretch_x, stretch_y, num_frames):
    """
    Creates a simulation with the bounding box as the polygon.
    """

    fig = plt.figure(0)
    ax = fig.add_subplot(111)

    points = helpers.generate_random_points(a, b, num_samples)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    vorinfo, boundary, bounding_box, centroids = test_utils.find_new_voronoi_points(points, 0, stretch_x=stretch_x, stretch_y=stretch_y, bound_box=True)

    for k in range(num_frames):
        plt.clf()
        vorinfo, _, _, centroids = test_utils.find_new_voronoi_points(points, bounding_box, stretch_x=stretch_x, stretch_y=stretch_y, bound_box=False)

        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        for region in vorinfo[0]:
            polygon_points = vorinfo[1][region]
            poly = Polygon(polygon_points).intersection(bounding_box)
            polygon = [p for p in poly.exterior.coords]
            plt.fill(*zip(*polygon), alpha=0.8, facecolor='none', edgecolor='black', linewidth=1)

        plt.plot(points_x, points_y, 'ko', markersize=3)
        plt.plot(centroids_x, centroids_y, 'bo', markersize=3)
        plt.xlim(np.ceil(boundary[0]), np.ceil(boundary[2]))
        plt.ylim(np.ceil(boundary[1]), np.ceil(boundary[3]))
        plt.axis('equal')
        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        plt.show(block=False)
        plt.pause(0.001)

def test_with_polygon(num_samples, num_frames):
    """
    Creates a simulation with a polygon.
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

    for k in range(num_frames):
        plt.clf()
        vorinfo, centroids = test_utils.find_new_voronoi_points_for_polygon(main_poly, points)

        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        for region in vorinfo[0]:
            polygon_points = vorinfo[1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            plt.fill(*zip(*polygon), alpha=0.8, facecolor='none', edgecolor='black', linewidth=1)

        plt.plot(points_x, points_y, 'ko', markersize=3)
        plt.plot(centroids_x, centroids_y, 'bo', markersize=3)
        plt.axis('equal')

        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        plt.show(block=False)
        plt.pause(0.1)

if __name__ == "__main__":

    test_with_bounding_box(a=0, b=20, num_samples=9, stretch_x=5, stretch_y=5, num_frames=10000)
    #test_with_polygon(num_samples=10, num_frames=1000)

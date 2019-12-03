"""
Python file to create a 'simulation' of the Voronoi diagrams.
"""

""" This file is depreciated. """

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
import matplotlib.animation as animation
from research_utils.vorutils import helpers

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim(-50, 50)
plt.ylim(-50, 50)

def find_new_voronoi_points(points, stretch_x=10, stretch_y=10, ani=True):
    """
    Given the points, compute the new Voronoi points, which are the centroids
    of the previous Voronoi partitions.
    """

    if ani is True:
        points_x = np.array([points[i][0][0] for i in range(len(points))])
        points_y = np.array([points[i][0][1] for i in range(len(points))])
    else:
        points_x = np.array([points[i][0] for i in range(len(points))])
        points_y = np.array([points[i][1] for i in range(len(points))])

    vor = Voronoi(points)
    regions, vertices = helpers.create_finite_voronoi_2d(vor)

    min_x = vor.min_bound[0] - stretch_x
    max_x = vor.max_bound[0] + stretch_x
    min_y = vor.min_bound[1] - stretch_y
    max_y = vor.max_bound[1] + stretch_y
    bounding_box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    # Find the centroids; these are the new points.
    centroids = list()

    for region in regions:
        polygon_points = vertices[region]
        poly = Polygon(polygon_points).intersection(bounding_box)
        centroids.append(list(poly.centroid.coords))
        polygon = [p for p in poly.exterior.coords]
        plt.fill(*zip(*polygon), alpha=0.8, facecolor='none', edgecolor='black', linewidth=1)

    centroids_x = np.array([centroids[i][0][0] for i in range(len(centroids))])
    centroids_y = np.array([centroids[i][0][1] for i in range(len(centroids))])

    return polygon, [points, points_x, points_y], [centroids, centroids_x, centroids_y], [min_x, max_x, min_y, max_y]

points = helpers.generate_random_points(0, 20, 10)
polygon, array_points, array_centroids, array_box = find_new_voronoi_points(points, ani=False)

def init():
    """
    Sets the initial configuration.
    """

    plt.plot(array_points[1], array_points[2], 'ko', markersize=3)
    plt.plot(array_centroids[1], array_centroids[2], 'bo', markersize=3)
    plt.axis('equal')

def animate(i, array_centroids):
    """
    Creates the animation.
    """
    array_centroids[0] = [[array_centroids[0][i][0][0], array_centroids[0][i][0][1]] for i in range(len(array_centroids[0]))]
    print(array_centroids[0])
    polygon, array_points, array_centroids, array_box = find_new_voronoi_points(array_centroids[0], ani=False)
    array_centroids[0] = [[array_centroids[0][i][0][0], array_centroids[0][i][0][1]] for i in range(len(array_centroids[0]))]
    print(array_centroids[0])
    plt.plot(array_points[1], array_points[2], 'ko', markersize=3)
    plt.plot(array_centroids[1], array_centroids[2], 'bo', markersize=3)
    plt.axis('equal')

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, fargs=(array_centroids,))
plt.show()

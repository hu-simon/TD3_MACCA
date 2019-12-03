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
from research_utils.vorutils import helpers

def find_new_voronoi_points(points, bounding_box, stretch_x=10, stretch_y=10, bound_box=False):
    """
    Given points, compute the new Voronoi points, which are the centroids
    of the previous Voronoi partitions.
    """

    vor = Voronoi(points)
    regions, vertices = helpers.create_finite_voronoi_2d(vor)

    min_x = vor.min_bound[0] - stretch_x
    max_x = vor.max_bound[0] + stretch_x
    min_y = vor.min_bound[1] - stretch_y
    max_y = vor.max_bound[1] + stretch_y

    if bound_box is True:
        bounding_box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    centroids = list()

    polygons = list()

    for region in regions:
        polygon_points = vertices[region]
        poly = Polygon(polygon_points).intersection(bounding_box)
        centroids.append(list(poly.centroid.coords))

    centroids = [[centroids[i][0][0], centroids[i][0][1]] for i in range(len(centroids))]

    return [regions, vertices], [min_x, min_y, max_x, max_y], bounding_box, centroids

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111)

    points = helpers.generate_random_points(0, 20, 10)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    vorinfo, boundary, bounding_box, centroids = find_new_voronoi_points(points, 0, stretch_x=10, stretch_y=10, bound_box=True)

    for k in range(5):
        vorinfo, _, _, centroids = find_new_voronoi_points(points, bounding_box, stretch_x=10, stretch_y=10, bound_box=False)

        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])\

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

        plt.show(block=True)

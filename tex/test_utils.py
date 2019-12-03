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

def find_new_voronoi_points_for_polygon(poly, points):
    """
    Given points and the enclosing polygon, compute the new Voronoi points which are taken
    to be the centroids of the previous Voronoi partitions.
    """

    vor = Voronoi(points)

    regions, vertices = helpers.create_finite_voronoi_2d(vor)

    centroids = list()

    polygons = list()

    for region in regions:
        polygon_points = vertices[region]
        polygon = Polygon(polygon_points).intersection(poly)
        centroids.append(list(polygon.centroid.coords))

    centroids = [[centroids[i][0][0], centroids[i][0][1]] for i in range(len(centroids))]

    return [regions, vertices], centroids

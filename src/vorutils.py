"""
Written by Simon Hu, all rights reserved.

Written for ECE 276C final project, University of California San Diego.
"""

import numpy as np
from shapely.ops import *
from scipy.spatial import *
from shapely.geometry import *

def random_initialization(poly, num_points, seed=2019):
    """
    Generates random tuples of the form (x,y) which lie within the interior of
    a polygon.

    Parameters
    ----------
    poly : Shapely object
        Shapely Polygon object representing the polygon used for determining
        membership.
    num_points : int
        Number of agents/points to generate.
    seed : int, optional
        Seed value for reproducing results.

    Returns
    -------
    random_points : list of tuples
        List containing tuples of the form (x,y) representing randomly
        sampled points that lie within the interior of a polygon.
    """
    random_points = list()
    min_x, min_y, max_x, max_y = poly.bounds

    while(len(random_points) < num_points):
        point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if(point.within(poly)):
            random_points.append(point.xy)

    # Convert it to the normal form.
    random_points = [[random_points[i][0][0], random_points[i][1][0]] for i in range(num_points)]
    return random_points

def compute_finite_voronoi(vor, radius=None):
    """
    Creates a finite Voronoi tesselation in 2D, given some previously computed
    infinite Voronoi tesselation.

    Parameters
    ----------
    vor : Scipy object
        Scipy geometric object representing the infinite Voronoi tesselation.
    radius : float, optional
        Float reprsenting the distance to 'the point at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each recomputed Voronoi tesselation.
    vertices : list of tuples
        Coordinates of the computed Voronoi regions.

    Notes
    -----
    * vertices contains the same coordinates returned by the scipy.spatial.Voronoi function
    except the points at infinity are appended to this list.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("2D input is required.")

    regions_new = list()
    vertices_new = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all of the ridges for a single point.
    ridges_all = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges_all.setdefault(p1, []).append((p2, v1, v2))
        ridges_all.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct the infinite regions.
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        # Reconstruct the finite regions.
        if all(v >= 0 for v in vertices):
            # If Qhull reports >=0 then this is a finite region, so we should add it to the list.
            regions_new.append(vertices)
            continue

        # Reconstruct the non-finite regions.
        ridges = ridges_all[p1]
        region_new = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # This is again a finite ridge and is already in the region.
                continue

            # Compute the missing endpoint for the infinite ridge.
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            point_far = vor.vertices[v2] + direction * radius

            region_new.append(len(vertices_new))
            vertices_new.append(point_far.tolist())

        # Sort the region counterclockwise.
        sorted_vertices = np.asarray([vertices_new[v] for v in region_new])
        c = sorted_vertices.mean(axis=0)
        angles = np.arctan2(sorted_vertices[:,1] - c[1], sorted_vertices[:,0] - c[0])
        region_new = np.array(region_new)[np.argsort(angles)]

        regions_new.append(region_new.tolist())
    return regions_new, np.asarray(vertices_new)

def compute_vorinfo(poly, points):
    """
    Given points and the enclosing polygon, computes the new Voronoi points
    which are taken to be the centroids of the previous Voronoi partitions.

    Parameters
    ----------
    poly : Shapely object
        Shapely geometric object representing the polygon of interest.

    points : list of tuples
        List of current points in the Voronoi tesselation.
    """
    vor = Voronoi(points)
    regions, vertices = compute_finite_voronoi(vor)
    centroids = list()
    polygons = list()

    for region in regions:
        polygon_points = vertices[region]
        polygon = Polygon(polygon_points).intersection(poly)
        centroids.append(list(polygon.centroid.coords))

    centroids = [[centroids[i][0][0], centroids[i][0][1]] for i in range(len(centroids))]

    return [regions, vertices], centroids

def compute_closest_point(poly, points):
    """
    Given a polygon, finds the closest point in points that is
    contained within the polygon.

    Parameters
    ----------
    poly : Shapely object
        Shapely geometric object representing the polygon of interest.
    points : list of tuples
        List of current points in the Voronoi tesselation.

    Returns
    -------
    closest_point : tuple
        The closest point contained in the polygon.
    idx_closest_point : int
        Index of the closest point contained in the polygon.
    """
    closest_point = None
    for idx, point in enumerate(points):
        if poly.contains(Point(point)):
            closest_point = point
            idx_closest_point = idx
        else:
            continue

    return closest_point, idx_closest_point

def create_transparent_cmap(cmap, N=255):
    """
    Copies the sepcified colormap and makes it a transparent one
    by playing around with the alpha values.

    Parameters
    ----------
    cmap : maplotlib cmap object
        Matplotlib cmap object representing the color map to make
        transparent.
    N : int, optional
        Integer used in determining the alpha value.

    Returns
    -------
    transparent_cmap : matplotlib cmap object
        Matplotlib cmap object representing the transparent cmap.
    """
    transparent_cmap = cmap
    transparent_cmap._init()
    transparent_cmap._lut[:,-1] = np.linspace(0,0.7,N+4)
    return transparent_cmap

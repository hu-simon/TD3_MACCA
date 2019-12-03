"""
Written by Simon Hu, all rights reserved.
"""

"""
TODO implement functionality that allows you to keep track of the total energy of the configuration.
This of course, requires that you integrate some cost function so...yeah
"""

import time
import test_utils
import numpy as np
from scipy.spatial import *
from shapely.geometry import *
import scipy.integrate as scint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from research_utils.vorutils import helpers
from research_utils.vorutils import calctools

def create_transparent_cmap(cmap, N=255):
    """
    Copies the specified colormap and makes it a transparent one
    by playing around with the alpha value.

    Parameters
    ----------
    cmap : matplotlib cmap object
        Matplotlib cmap object representing the color map to make
        transparent.
    N : int, optional
        Integer used in determining the alpha value.

    Returns
    -------
    transparent_cmap : matplotlib cmap object
        Matplotlib cmap object representing the transparent
        cmap.
    """
    transparent_cmap = cmap
    transparent_cmap._init()
    transparent_cmap._lut[:,-1] = np.linspace(0, 0.5, N+4)
    return transparent_cmap

def simple_dynamics(x, u):
    """
    Integrator dynamics.

    Parameters
    ----------
    x : numpy array
        Numpy array representing the current state.
    u : numpy array
        Numpy array representing the control input.

    Returns
    -------
    dynamics : numpy array
        Numpy array representing the dynamics of the system.
    """
    return np.stack((u[0], u[1]))

def rk4(func, x, u, dt=0.1):
    """
    Performs Runge-Kutta 4-th order integration to find the next point.

    Parameters
    ----------
    func : python lambda function
        Python lambda function representing the dyanmics of the system.
    x : numpy array
        Numpy array representing the current state.
    u : numpy array
        Numpy array representing the control input.
    dt : double, optional
        Step size of the numerical scheme.

    Returns
    -------
    new_pos : numpy array
        Numpy array representing the new position of the Agent.
    """
    x = np.array(x)
    u = np.array(u)
    k1 = func(x + (dt/2), u)
    k2 = func(x + (dt/2) * k1, u)
    k3 = func(x + (dt/2) * k2, u)
    k4 = func(x + dt * k3, u)
    new_pos = (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_pos

def feedback_controller(gain, curr_pos, next_pos):
    """
    Creates a feedback controller that drives the position
    of the Agent to the centroid.

    Parameters
    ----------
    gain : double
        The gain of the controller.
    curr_pos : numpy array
        Numpy array representing the current position of the Agent.
    next_pos : numpy array
        Numpy array representing the next position of the Agent. This is
        the position we want to drive the Agent to.

    Returns
    -------
    u : numpy array
        Numpy array representing the control input to put into
        the system.
    """
    u = gain * (np.array(next_pos) - np.array(curr_pos))
    return u

def track_gaussian_centroid(num_samples=10, num_frames=100):
    """
    Tracks the location of the centroids as the Voronoi diagram changes with time.
    This time, we take into account the weighted centroid locations.

    Parameters
    ----------
    num_samples : int, optional
        Number of agents to populate the environment with.
    num_frames : int, optional
        Number of frames for the simulation.
    """
    fig = plt.figure(0)
    ax = fig.add_subplot(111)

    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)

    min_x, min_y, max_x, max_y = main_poly.bounds

    """ Code below tests points that are fractioned apart from each other. """

    """
    sample_bounds_1 = [[20,20], [40,20], [20,40], [40,40]]
    sample_bounds_2 = [[-20,-20], [-40,-20], [-20,-40], [-40,-40]]
    sample_poly_1 = Polygon(sample_bounds_1)
    sample_poly_2 = Polygon(sample_bounds_2)

    points1 = helpers.generate_points_within_polygon(sample_poly_1, num_points=num_samples)
    points2 = helpers.generate_points_within_polygon(sample_poly_2, num_points=num_samples)
    points = points1 + points2
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])
    """

    """ Code below starts the Agents on the bottom side of the pentagon. """

    sample_bounds = [[-50,-70], [50,-70], [-50,-50], [50, -50]]
    sample_poly = Polygon(sample_bounds)
    points = helpers.generate_points_within_polygon(sample_poly, num_points=num_samples)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    start_points = points.copy()
    start_points_x = points_x.copy()
    start_points_y = points_y.copy()

    ext_x, ext_y = main_poly.exterior.xy
    plt.plot(ext_x, ext_y, color='black', linewidth=0.7)

    mu_x    = 0
    mu_y    = 50
    sigma_x = 20
    sigma_y = 20

    mu2_x    = 0
    mu2_y    = -50
    sigma2_x = 20
    sigma2_y = 20

    mu3_x    = 0
    mu3_y    = 0
    sigma3_x = 20
    sigma3_y = 20

    """ Code below handles the case for only one Gaussian. """
    """
    joint_xy = lambda x,y : (1/(2 * np.pi * sigma_x * sigma_y)) * \
               np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2)))
    joint_x = lambda x : (1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2)))
    joint_y = lambda y : (1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2)))
    weighted_x = lambda x : x * (1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2)))
    weighted_y = lambda y : y * (1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2)))
    """

    """ Code below handles the case for two Gaussians. """
    """
    joint_xy = lambda x,y : (1/(2 * np.pi * sigma_x * sigma_y)) * \
               np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2))) \
               + (1/(2 * np.pi * sigma2_x * sigma2_y)) * \
               np.exp(-0.5 * (((x - mu2_x)**2)/(2 * sigma2_x**2) + ((y - mu2_y)**2)/(2 * sigma2_y**2)))
    joint_x = lambda x : (1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))) \
              + (1/(2 * np.pi * sigma2_x)) * np.exp(-0.5 * ((x - mu2_x)**2/(2 * sigma2_x**2)))
    joint_y = lambda y : (1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))) \
              + (1/(2 * np.pi * sigma2_y)) * np.exp(-0.5 * ((y - mu2_y)**2/(2 * sigma2_y**2)))
    weighted_x = lambda x : x * ((1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))) \
                            + (1/(2 * np.pi * sigma2_x)) * np.exp(-0.5 * ((x - mu2_x)**2/(2 * sigma2_x**2))))
    weighted_y = lambda y : y * ((1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))) \
                            + (1/(2 * np.pi * sigma2_y)) * np.exp(-0.5 * ((y - mu2_y)**2/(2 * sigma2_y**2))))
    """
    """ Code below handles the case for three Gaussians. It is not done. """

    joint_xy = lambda x,y : (1/(2 * np.pi * sigma_x * sigma_y)) * \
               np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2))) \
               + (1/(2 * np.pi * sigma2_x * sigma2_y)) * \
               np.exp(-0.5 * (((x - mu2_x)**2)/(2 * sigma2_x**2) + ((y - mu2_y)**2)/(2 * sigma2_y**2))) \
               + (1/(2 * np.pi * sigma3_x * sigma3_y)) * \
               np.exp(-0.5 * (((x - mu3_x)**2)/(2 * sigma3_x**2) + ((y - mu3_y)**2)/(2 * sigma3_y**2)))
    joint_x = lambda x : (1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))) \
              + (1/(2 * np.pi * sigma2_x)) * np.exp(-0.5 * ((x - mu2_x)**2/(2 * sigma2_x**2))) \
              + (1/(2 * np.pi * sigma3_x)) * np.exp(-0.5 * ((x - mu3_x)**2/(2 * sigma3_x**2)))
    joint_y = lambda y : (1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))) \
              + (1/(2 * np.pi * sigma2_y)) * np.exp(-0.5 * ((y - mu2_y)**2/(2 * sigma2_y**2))) \
              + (1/(2 * np.pi * sigma3_y)) * np.exp(-0.5 * ((y - mu3_y)**2/(2 * sigma3_y**2)))
    weighted_x = lambda x : x * ((1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))) \
                            + (1/(2 * np.pi * sigma2_x)) * np.exp(-0.5 * ((x - mu2_x)**2/(2 * sigma2_x**2))) \
                            + (1/(2 * np.pi * sigma3_x)) * np.exp(-0.5 * ((x - mu3_x)**2/(2 * sigma3_x**2))))
    weighted_y = lambda y : y * ((1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))) \
                            + (1/(2 * np.pi * sigma2_y)) * np.exp(-0.5 * ((y - mu2_y)**2/(2 * sigma2_y**2))) \
                            + (1/(2 * np.pi * sigma3_y)) * np.exp(-0.5 * ((y - mu3_y)**2/(2 * sigma3_y**2))))


    cost_list = list()
    for k in range(num_frames):
        vorinfo, _ = test_utils.find_new_voronoi_points_for_polygon(main_poly, points)
        centroids = list()
        total_cost = 0
        # Compute the contribution to the total cost function.
        for region in vorinfo[0]:
            polygon_points = vorinfo[1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            min_x, min_y, max_x, max_y = poly.bounds

            # Find which p_i is contained in the polygon.
            p_i = None
            for point in points:
                if poly.contains(Point(point)):
                    p_i = point
                else:
                    continue

            mass_x, _ = scint.quad(joint_x, min_x, max_x)
            mass_y, _ = scint.quad(joint_y, min_y, max_y)
            total_mass = mass_x * mass_y

            # Compute the centroids.
            cent_x, _ = scint.quad(weighted_x, min_x, max_x)
            cent_x = (1 / mass_x) * cent_x
            cent_y, _ = scint.quad(weighted_y, min_y, max_y)
            cent_y = (1 / mass_y) * cent_y

            """ Code below uses MY integration scheme.
            mass_x, _ = calctools.gl_quadrature(joint_x, min_x, max_x, ord=5)
            mass_y, _ = calctools.gl_quadrature(joint_y, min_y, max_y, ord=5)
            total_mass = mass_x * mass_y

            cent_x, _ = calctools.gl_quadrature(weighted_x, min_x, max_x, ord=5)
            cent_x = (1 / mass_x) * cent_x
            cent_y, _ = calctools.gl_quadrature(weighted_y, min_y, max_y, ord=5)
            cent_y = (1 / mass_y) * cent_y
            """
            centroids.append([cent_x, cent_y])

            cost_function_x = lambda x : -(x - p_i[0])**2 * joint_x(x)
            cost_function_y = lambda y : -(y - p_i[1])**2 * joint_y(y)

            total_cost_x, _ = scint.quad(cost_function_x, min_x, max_x)
            total_cost_y, _ = scint.quad(cost_function_y, min_y, max_y)
            total_cost += total_cost_x + total_cost_y
        print(total_cost)
        cost_list.append(total_cost)
        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        if (k == 0):
            plt.plot(points_x, points_y, 'r*', markersize=1)
        else:
            plt.plot(centroids_x, centroids_y, 'bo', markersize=0.3)
        plt.axis('equal')

        """ Code below checks the cost function and stops the program if it drops below some threshold. """


        """ Code below stops the program once the total change in the Agent positions drops below some threshold. """
        """
        print("{}: {}".format(k, np.sum(np.abs(np.array(centroids) - np.array(points))**2)))
        # Check to see if the change in centroid positions for all of them are small enough.
        if (np.sum(np.abs(np.array(centroids) - np.array(points))**2) < .01):
            print('Breaking early')
            for region in vorinfo[0]:
                polygon_points = vorinfo[1][region]
                poly = Polygon(polygon_points).intersection(main_poly)
                polygon = [p for p in poly.exterior.coords]
                plt.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.7)
            # Create a final diagram with the final position of the
            min_x, max_x, min_y, max_y = plt.axis()
            fig, ax = plt.subplots(1,1)
            ax.plot(points_x, points_y, 'bo', markersize=1)
            ax.plot(start_points_x, start_points_y, 'r*', markersize=1)
            ax.axis('equal')
            for region in vorinfo[0]:
                polygon_points = vorinfo[1][region]
                poly = Polygon(polygon_points).intersection(main_poly)
                polygon = [p for p in poly.exterior.coords]
                ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.7)

            transparent_cmap = create_transparent_cmap(plt.cm.Greens)
            # Plot the Gaussian kernel.
            x, y = np.mgrid[min_x:max_x, min_y:max_y]
            gauss_val = joint_xy(x,y).ravel()
            cb = ax.contourf(x,y,gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
            plt.colorbar(cb)
            break
        """
        # Update the points (e.g. this is where you can include the controller update).
        """ Code below implements the teleportation dynamics. """
        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        """ Code below attempts to implement my controller. It does not work. """
        """
        new_points = list()
        for idx, point in enumerate(points):
            u = feedback_controller(0.1, point, centroids[idx])
            new_point = rk4(simple_dynamics, point, u, dt=0.01)
            new_points.append(new_point)

        points = new_points.copy()
        points_x = np.array([new_points[i][0] for i in range(len(new_points))])
        points_y = np.array([new_points[i][1] for i in range(len(new_points))])
        """

        if k == num_frames-1:
            # Draw the final Voronoi configuration.
            for region in vorinfo[0]:
                polygon_points = vorinfo[1][region]
                poly = Polygon(polygon_points).intersection(main_poly)
                polygon = [p for p in poly.exterior.coords]
                plt.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.7)

            # Create a final diagram with the final position of the
            """
            plt.figure(1)
            plt.plot(points_x, points_y, 'bo', markersize=1)
            plt.plot(start_points_x, start_points_y, 'r*', markersize=1)
            plt.axis('equal')
            for region in vorinfo[0]:
                polygon_points = vorinfo[1][region]
                poly = Polygon(polygon_points).intersection(main_poly)
                polygon = [p for p in poly.exterior.coords]
                plt.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.7)
            # Plot some Gaussian kernel.
            """
            min_x, max_x, min_y, max_y = plt.axis()
            fig, ax = plt.subplots(1,1)
            ax.plot(points_x, points_y, 'bo', markersize=1)
            ax.plot(start_points_x, start_points_y, 'r*', markersize=1)
            ax.axis('equal')
            for region in vorinfo[0]:
                polygon_points = vorinfo[1][region]
                poly = Polygon(polygon_points).intersection(main_poly)
                polygon = [p for p in poly.exterior.coords]
                ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.7)

            transparent_cmap = create_transparent_cmap(plt.cm.Greens)
            # Plot the Gaussian kernel.
            x, y = np.mgrid[min_x:max_x, min_y:max_y]
            gauss_val = joint_xy(x,y).ravel()
            cb = ax.contourf(x,y,gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
            plt.colorbar(cb)

            plt.figure(2)
            plt.plot(-1 * np.array(cost_list[1:len(cost_list)]))
        plt.show(block=False)
        plt.pause(0.1)
    plt.show(block=True)
    #plt.savefig('gaussian_simulation_3.png')


def test_gaussian_voronoi_movie(num_samples=5, num_frames=100):
    pass

if __name__ == "__main__":
    track_gaussian_centroid(num_samples=10, num_frames=100)

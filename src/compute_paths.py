"""
Written by Simon Hu, all rights reserved.

Written for ECE276C, University of California San Diego.
"""

import time
import vorutils
import numpy as np
from scipy.spatial import *
from shapely.geometry import *
import scipy.integrate as scint
import matplotlib.pyplot as plt

num_frames = 200
num_agents = 20
agent_max_range = 1

def scenario_one():
    """
    Computes the paths for scenario one.
    """
    # Simulation parameters.

    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)

    min_x, min_y, max_x, max_y = main_poly.bounds

    sample_bounds = [[-50*0.5,-70*0.5], [50*0.5,-70*0.5], [-50*0.5,-50*0.5], [50*0.5, -50*0.5]]
    sample_poly = Polygon(sample_bounds)
    points = vorutils.random_initialization(sample_poly, num_points=num_agents, seed=2020)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    mu_x    = -50
    mu_y    = 0
    sigma_x = 20
    sigma_y = 20

    joint_xy = lambda x,y : (1 / (2 * np.pi * sigma_x * sigma_y)) * \
               np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2)))

    joint_x = lambda x : (1 / (2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2 / (2 * sigma_x**2)))

    joint_y = lambda y : (1 / (2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2 / (2 * sigma_y**2)))

    weighted_x = lambda x : x * ((1 / (2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))))

    weighted_y = lambda y : y * (( 1 /(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))))

    cost_list = list()
    path_array = np.zeros((num_frames, num_agents, 2))
    vorlist = list()

    # Add the initial point.
    for i in range(num_agents):
        path_array[0,i,0] = points_x[i]
        path_array[0,i,1] = points_y[i]

    for k in range(num_frames):
        vorinfo, _ = vorutils.compute_vorinfo(main_poly, points)
        vorlist.append(vorinfo)
        centroids = list()
        total_cost = 0

        for region in vorinfo[0]:
            polygon_points = vorinfo[1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            min_x, min_y, max_x, max_y = poly.bounds

            closest_point, _ = vorutils.compute_closest_point(poly, points)
            mass_x, _ = scint.quad(joint_x, min_x, max_x)
            mass_y, _ = scint.quad(joint_y, min_y, max_y)
            total_mass = mass_x + mass_y

            # Compute the centroids
            cent_x, _ = scint.quad(weighted_x, min_x, max_x)
            cent_x = (1 / mass_x) * cent_x
            cent_y, _ = scint.quad(weighted_y, min_y, max_y)
            cent_y = (1 / mass_y) * cent_y

            if (np.linalg.norm(np.array([cent_x, cent_y]) - np.array([closest_point[0], closest_point[1]])) > agent_max_range**2):
                centroids.append([closest_point[0] + (np.sign(cent_x - closest_point[0]) + np.random.randn()) * agent_max_range, closest_point[1] + (np.sign(cent_y - closest_point[1]) + np.random.randn()) * agent_max_range])
            else:
                centroids.append([cent_x, cent_y])

            cost_function_x = lambda x : -(x - closest_point[0])**2 * joint_x(x)
            cost_function_y = lambda y : -(y - closest_point[1])**2 * joint_y(y)

            total_cost_x, _ = scint.quad(cost_function_x, min_x, max_x)
            total_cost_y, _ = scint.quad(cost_function_y, min_y, max_y)
            total_cost += total_cost_x + total_cost_y

        cost_list.append(total_cost)
        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        for i in range(num_agents):
            path_array[k,i,0] = points_x[i]
            path_array[k,i,1] = points_y[i]

    np.save('scenario_one.npy', path_array)
    np.save('cost_scenario_one.npy', np.array(cost_list))
    np.save('vorinfo_scenario_one.npy', np.array(vorlist), allow_pickle=True)

def scenario_two():
    """
    Computes the path for scenario two.
    """
    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)

    min_x, min_y, max_x, max_y = main_poly.bounds

    sample_bounds = [[-50*0.5,-70*0.5], [50*0.5,-70*0.5], [-50*0.5,-50*0.5], [50*0.5, -50*0.5]]
    sample_poly = Polygon(sample_bounds)
    points = vorutils.random_initialization(sample_poly, num_points=num_agents)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    mu_x     = -50
    mu_y     = 0
    sigma_x  = 20
    sigma_y  = 20

    mu2_x    = -40
    mu2_y    = 40
    sigma2_x = 20
    sigma2_y = 20

    mu3_x     = -40
    mu3_y     = -40
    sigma3_x  = 20
    sigma3_y  = 20

    mu4_x    = 40
    mu4_y    = 40
    sigma4_x = 20
    sigma4_y = 20

    mu5_x    = 40
    mu5_y    = -40
    sigma5_x = 20
    sigma5_y = 20

    joint_xy = lambda x,y : (1/(2 * np.pi * sigma_x * sigma_y)) * \
               np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2))) \
               + (1/(2 * np.pi * sigma2_x * sigma2_y)) * \
               np.exp(-0.5 * (((x - mu2_x)**2)/(2 * sigma2_x**2) + ((y - mu2_y)**2)/(2 * sigma2_y**2))) \
               + (1/(2 * np.pi * sigma3_x * sigma3_y)) * \
               np.exp(-0.5 * (((x - mu3_x)**2)/(2 * sigma3_x**2) + ((y - mu3_y)**2)/(2 * sigma3_y**2))) \
               + (1/(2 * np.pi * sigma4_x * sigma4_y)) * \
               np.exp(-0.5 * (((x - mu4_x)**2)/(2 * sigma4_x**2) + ((y - mu4_y)**2)/(2 * sigma4_y**2))) \
               + (1/(2 * np.pi * sigma5_x * sigma5_y)) * \
               np.exp(-0.5 * (((x - mu5_x)**2)/(2 * sigma5_x**2) + ((y - mu5_y)**2)/(2 * sigma5_y**2)))
    joint_x = lambda x : (1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))) \
              + (1/(2 * np.pi * sigma2_x)) * np.exp(-0.5 * ((x - mu2_x)**2/(2 * sigma2_x**2))) \
              + (1/(2 * np.pi * sigma3_x)) * np.exp(-0.5 * ((x - mu3_x)**2/(2 * sigma3_x**2))) \
              + (1/(2 * np.pi * sigma4_x)) * np.exp(-0.5 * ((x - mu4_x)**2/(2 * sigma4_x**2))) \
              + (1/(2 * np.pi * sigma5_x)) * np.exp(-0.5 * ((x - mu5_x)**2/(2 * sigma5_x**2)))
    joint_y = lambda y : (1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))) \
              + (1/(2 * np.pi * sigma2_y)) * np.exp(-0.5 * ((y - mu2_y)**2/(2 * sigma2_y**2))) \
              + (1/(2 * np.pi * sigma3_y)) * np.exp(-0.5 * ((y - mu3_y)**2/(2 * sigma3_y**2))) \
              + (1/(2 * np.pi * sigma4_y)) * np.exp(-0.5 * ((y - mu4_y)**2/(2 * sigma4_y**2))) \
              + (1/(2 * np.pi * sigma5_y)) * np.exp(-0.5 * ((y - mu5_y)**2/(2 * sigma5_y**2)))
    weighted_x = lambda x : x * ((1/(2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))) \
                            + (1/(2 * np.pi * sigma2_x)) * np.exp(-0.5 * ((x - mu2_x)**2/(2 * sigma2_x**2))) \
                            + (1/(2 * np.pi * sigma3_x)) * np.exp(-0.5 * ((x - mu3_x)**2/(2 * sigma3_x**2))) \
                            + (1/(2 * np.pi * sigma4_x)) * np.exp(-0.5 * ((x - mu4_x)**2/(2 * sigma4_x**2))) \
                            + (1/(2 * np.pi * sigma5_x)) * np.exp(-0.5 * ((x - mu5_x)**2/(2 * sigma5_x**2))))
    weighted_y = lambda y : y * ((1/(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))) \
                            + (1/(2 * np.pi * sigma2_y)) * np.exp(-0.5 * ((y - mu2_y)**2/(2 * sigma2_y**2))) \
                            + (1/(2 * np.pi * sigma3_y)) * np.exp(-0.5 * ((y - mu3_y)**2/(2 * sigma3_y**2))) \
                            + (1/(2 * np.pi * sigma4_y)) * np.exp(-0.5 * ((y - mu4_y)**2/(2 * sigma4_y**2))) \
                            + (1/(2 * np.pi * sigma5_y)) * np.exp(-0.5 * ((y - mu5_y)**2/(2 * sigma5_y**2))))
    cost_list = list()
    path_array = np.zeros((num_frames, num_agents, 2))
    vorlist = list()

    # Add the initial point.
    for i in range(num_agents):
        path_array[0,i,0] = points_x[i]
        path_array[0,i,1] = points_y[i]

    for k in range(num_frames):
        vorinfo, _ = vorutils.compute_vorinfo(main_poly, points)
        vorlist.append(vorinfo)
        centroids = list()
        total_cost = 0

        for region in vorinfo[0]:
            polygon_points = vorinfo[1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            min_x, min_y, max_x, max_y = poly.bounds

            closest_point, _ = vorutils.compute_closest_point(poly, points)
            mass_x, _ = scint.quad(joint_x, min_x, max_x)
            mass_y, _ = scint.quad(joint_y, min_y, max_y)
            total_mass = mass_x + mass_y

            # Compute the centroids
            cent_x, _ = scint.quad(weighted_x, min_x, max_x)
            cent_x = (1 / mass_x) * cent_x
            cent_y, _ = scint.quad(weighted_y, min_y, max_y)
            cent_y = (1 / mass_y) * cent_y

            if (np.linalg.norm(np.array([cent_x, cent_y]) - np.array([closest_point[0], closest_point[1]])) > agent_max_range**2):
                centroids.append([closest_point[0] + (np.sign(cent_x - closest_point[0]) + np.random.randn()) * agent_max_range, closest_point[1] + (np.sign(cent_y - closest_point[1]) + np.random.randn()) * agent_max_range])
            else:
                centroids.append([cent_x, cent_y])

            cost_function_x = lambda x : -(x - closest_point[0])**2 * joint_x(x)
            cost_function_y = lambda y : -(y - closest_point[1])**2 * joint_y(y)

            total_cost_x, _ = scint.quad(cost_function_x, min_x, max_x)
            total_cost_y, _ = scint.quad(cost_function_y, min_y, max_y)
            total_cost += total_cost_x + total_cost_y

        cost_list.append(total_cost)
        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        for i in range(num_agents):
            path_array[k,i,0] = points_x[i]
            path_array[k,i,1] = points_y[i]

    np.save('scenario_two.npy', path_array)
    np.save('cost_scenario_two.npy', np.array(cost_list))
    np.save('vorinfo_scenario_two.npy', np.array(vorlist), allow_pickle=True)

def scenario_three():
    """
    Computes the path for scenario three.
    """
    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)

    min_x, min_y, max_x, max_y = main_poly.bounds

    sample_bounds = [[-50*0.5,-70*0.5], [50*0.5,-70*0.5], [-50*0.5,-50*0.5], [50*0.5, -50*0.5]]
    sample_poly = Polygon(sample_bounds)
    points = vorutils.random_initialization(sample_poly, num_points=num_agents)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    mu_x     = 0
    mu_y     = 50
    sigma_x  = 10
    sigma_y  = 30

    mu2_x    = 0
    mu2_y    = 0
    sigma2_x = 10
    sigma2_y = 30

    mu3_x     = 0
    mu3_y     = -50
    sigma3_x  = 10
    sigma3_y  = 30

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
    path_array = np.zeros((num_frames, num_agents, 2))
    vorlist = list()

    # Add the initial point.
    for i in range(num_agents):
        path_array[0,i,0] = points_x[i]
        path_array[0,i,1] = points_y[i]

    for k in range(num_frames):
        vorinfo, _ = vorutils.compute_vorinfo(main_poly, points)
        vorlist.append(vorinfo)
        centroids = list()
        total_cost = 0

        for region in vorinfo[0]:
            polygon_points = vorinfo[1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            min_x, min_y, max_x, max_y = poly.bounds

            closest_point, _ = vorutils.compute_closest_point(poly, points)
            mass_x, _ = scint.quad(joint_x, min_x, max_x)
            mass_y, _ = scint.quad(joint_y, min_y, max_y)
            total_mass = mass_x + mass_y

            # Compute the centroids
            cent_x, _ = scint.quad(weighted_x, min_x, max_x)
            cent_x = (1 / mass_x) * cent_x
            cent_y, _ = scint.quad(weighted_y, min_y, max_y)
            cent_y = (1 / mass_y) * cent_y

            if (np.linalg.norm(np.array([cent_x, cent_y]) - np.array([closest_point[0], closest_point[1]])) > agent_max_range**2):
                centroids.append([closest_point[0] + (np.sign(cent_x - closest_point[0]) + np.random.randn()) * agent_max_range, closest_point[1] + (np.sign(cent_y - closest_point[1]) + np.random.randn()) * agent_max_range])
            else:
                centroids.append([cent_x, cent_y])

            cost_function_x = lambda x : -(x - closest_point[0])**2 * joint_x(x)
            cost_function_y = lambda y : -(y - closest_point[1])**2 * joint_y(y)

            total_cost_x, _ = scint.quad(cost_function_x, min_x, max_x)
            total_cost_y, _ = scint.quad(cost_function_y, min_y, max_y)
            total_cost += total_cost_x + total_cost_y

        cost_list.append(total_cost)
        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        for i in range(num_agents):
            path_array[k,i,0] = points_x[i]
            path_array[k,i,1] = points_y[i]

    np.save('scenario_three.npy', path_array)
    np.save('cost_scenario_three.npy', np.array(cost_list))
    np.save('vorinfo_scenario_three.npy', np.array(vorlist), allow_pickle=True)

def scenario_four(num_frames=1000):
    """
    Computes the path for scenario four.
    """
    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)

    min_x, min_y, max_x, max_y = main_poly.bounds

    sample_bounds = [[-50*0.5,-70*0.5], [50*0.5,-70*0.5], [-50*0.5,-50*0.5], [50*0.5, -50*0.5]]
    sample_poly = Polygon(sample_bounds)
    points = vorutils.random_initialization(sample_poly, num_points=num_agents)
    points_x = np.array([points[i][0] for i in range(len(points))])
    points_y = np.array([points[i][1] for i in range(len(points))])

    cost_list = list()
    path_array = np.zeros((num_frames, num_agents, 2))
    vorlist = list()

    # Add the initial point.
    for i in range(num_agents):
        path_array[0,i,0] = points_x[i]
        path_array[0,i,1] = points_y[i]

    for k in range(num_frames):
        if k < 300:
            mu    = [0, 75]
            sigma = [20, 20]

            joint_xy = lambda x,y : (1 / (2 * np.pi * sigma[0] * sigma[1])) * \
                       np.exp(-0.5 * (((x - mu[0])**2)/(2 * sigma[0]**2) + ((y - mu[1])**2)/(2 * sigma[1]**2)))

            joint_x = lambda x : (1 / (2 * np.pi * sigma[0])) * np.exp(-0.5 * ((x - mu[0])**2 / (2 * sigma[0]**2)))

            joint_y = lambda y : (1 / (2 * np.pi * sigma[1])) * np.exp(-0.5 * ((y - mu[1])**2 / (2 * sigma[1]**2)))

            weighted_x = lambda x : x * ((1 / (2 * np.pi * sigma[0])) * np.exp(-0.5 * ((x - mu[0])**2/(2 * sigma[0]**2))))

            weighted_y = lambda y : y * (( 1 /(2 * np.pi * sigma[1])) * np.exp(-0.5 * ((y - mu[1])**2/(2 * sigma[1]**2))))
        elif k >= 300 and k < 600:
            mu     = [-40,-40]
            mu2    = [40,-40]
            sigma  = [20,20]
            sigma2 = [20,20]

            joint_xy = lambda x,y : (1/(2 * np.pi * sigma[0] * sigma[0])) * \
                       np.exp(-0.5 * (((x - mu[0])**2)/(2 * sigma[0]**2) + ((y - mu[1])**2)/(2 * sigma[1]**2))) \
                       + (1/(2 * np.pi * sigma2[0] * sigma2[1])) * \
                       np.exp(-0.5 * (((x - mu2[0])**2)/(2 * sigma2[0]**2) + ((y - mu2[1])**2)/(2 * sigma2[1]**2)))
            joint_x = lambda x : (1/(2 * np.pi * sigma[0])) * np.exp(-0.5 * ((x - mu[0])**2/(2 * sigma[0]**2))) \
                      + (1/(2 * np.pi * sigma2[0])) * np.exp(-0.5 * ((x - mu2[0])**2/(2 * sigma2[0]**2)))
            joint_y = lambda y : (1/(2 * np.pi * sigma[1])) * np.exp(-0.5 * ((y - mu[1])**2/(2 * sigma[1]**2))) \
                      + (1/(2 * np.pi * sigma2[1])) * np.exp(-0.5 * ((y - mu2[1])**2/(2 * sigma2[1]**2)))
            weighted_x = lambda x : x * ((1/(2 * np.pi * sigma[0])) * np.exp(-0.5 * ((x - mu[0])**2/(2 * sigma[0]**2))) \
                                    + (1/(2 * np.pi * sigma2[0])) * np.exp(-0.5 * ((x - mu2[0])**2/(2 * sigma2[0]**2))))
            weighted_y = lambda y : y * ((1/(2 * np.pi * sigma[1])) * np.exp(-0.5 * ((y - mu[1])**2/(2 * sigma[1]**2))) \
                                    + (1/(2 * np.pi * sigma2[1])) * np.exp(-0.5 * ((y - mu2[1])**2/(2 * sigma2[1]**2))))

        else:
            mu_x    = 0
            mu_y    = 0
            sigma_x = 20
            sigma_y = 20

            joint_xy = lambda x,y : (1 / (2 * np.pi * sigma_x * sigma_y)) * \
                       np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2)))

            joint_x = lambda x : (1 / (2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2 / (2 * sigma_x**2)))

            joint_y = lambda y : (1 / (2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2 / (2 * sigma_y**2)))

            weighted_x = lambda x : x * ((1 / (2 * np.pi * sigma_x)) * np.exp(-0.5 * ((x - mu_x)**2/(2 * sigma_x**2))))

            weighted_y = lambda y : y * (( 1 /(2 * np.pi * sigma_y)) * np.exp(-0.5 * ((y - mu_y)**2/(2 * sigma_y**2))))


        vorinfo, _ = vorutils.compute_vorinfo(main_poly, points)
        vorlist.append(vorinfo)
        centroids = list()
        total_cost = 0

        for region in vorinfo[0]:
            polygon_points = vorinfo[1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            min_x, min_y, max_x, max_y = poly.bounds

            closest_point, _ = vorutils.compute_closest_point(poly, points)
            mass_x, _ = scint.quad(joint_x, min_x, max_x)
            mass_y, _ = scint.quad(joint_y, min_y, max_y)
            total_mass = mass_x + mass_y

            # Compute the centroids
            cent_x, _ = scint.quad(weighted_x, min_x, max_x)
            cent_x = (1 / mass_x) * cent_x
            cent_y, _ = scint.quad(weighted_y, min_y, max_y)
            cent_y = (1 / mass_y) * cent_y

            if (np.linalg.norm(np.array([cent_x, cent_y]) - np.array([closest_point[0], closest_point[1]])) > agent_max_range**2):
                centroids.append([closest_point[0] + (np.sign(cent_x - closest_point[0]) + np.random.randn()) * agent_max_range, closest_point[1] + (np.sign(cent_y - closest_point[1]) + np.random.randn()) * agent_max_range])
            else:
                centroids.append([cent_x, cent_y])

            cost_function_x = lambda x : -(x - closest_point[0])**2 * joint_x(x)
            cost_function_y = lambda y : -(y - closest_point[1])**2 * joint_y(y)

            total_cost_x, _ = scint.quad(cost_function_x, min_x, max_x)
            total_cost_y, _ = scint.quad(cost_function_y, min_y, max_y)
            total_cost += total_cost_x + total_cost_y

        cost_list.append(total_cost)
        centroids_x = np.array([centroids[i][0] for i in range(len(centroids))])
        centroids_y = np.array([centroids[i][1] for i in range(len(centroids))])

        points = centroids.copy()
        points_x = centroids_x.copy()
        points_y = centroids_y.copy()

        for i in range(num_agents):
            path_array[k,i,0] = points_x[i]
            path_array[k,i,1] = points_y[i]

        np.save('scenario_four.npy', path_array)
        np.save('cost_scenario_four.npy', np.array(cost_list))
        np.save('vorinfo_scenario_four.npy', np.array(vorlist), allow_pickle=True)

if __name__ == "__main__":
    scenario_one()
    scenario_two()
    scenario_three()
    scenario_four(num_frames=1000)

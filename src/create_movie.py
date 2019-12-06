"""
Written by Simon Hu, all rights reserved.

Written for ECE 276C final project, University of California San Diego.
"""

import vorutils
import numpy as np
from shapely.ops import *
from scipy.spatial import *
from shapely.geometry import *
import scipy.integrate as scint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import vorutils
import numpy as np
import matplotlib.pyplot as plt

def choose_function(scenario, idx=0):
    """
    Chooses the lambda function depending on which scenario we are executing.

    Parameters
    ----------
    scenario : String
        String representing the type of scenario we are executing.

    Returns
    -------
    func : Python lambda function
        A python lambda function representing the joint density.
    """
    if scenario is "one":
        mu    = [-50, 0]
        sigma = [20, 20]
        func = lambda x,y : (1/(2 * np.pi * sigma[0] * sigma[1])) * \
               np.exp(-0.5 * (((x - mu[0])**2)/(2 * sigma[0]**2) + ((y - mu[1])**2)/(2 * sigma[1]**2)))
    elif scenario is "two":
        mu, mu2, mu3, mu4, mu5                = [-50,0], [-40,40], [-40,-40], [40,40], [40,-40]
        sigma, sigma2, sigma3, sigma4, sigma5 = [20,20], [20,20], [20,20], [20,20], [20,20]
        func = lambda x,y : (1/(2 * np.pi * sigma[0] * sigma[1])) * \
               np.exp(-0.5 * (((x - mu[0])**2)/(2 * sigma[0]**2) + ((y - mu[1])**2)/(2 * sigma[1]**2))) \
               + (1/(2 * np.pi * sigma2[0] * sigma2[1])) * \
               np.exp(-0.5 * (((x - mu2[0])**2)/(2 * sigma2[0]**2) + ((y - mu2[1])**2)/(2 * sigma2[1]**2))) \
               + (1/(2 * np.pi * sigma3[0] * sigma3[1])) * \
               np.exp(-0.5 * (((x - mu3[0])**2)/(2 * sigma3[0]**2) + ((y - mu3[1])**2)/(2 * sigma3[1]**2))) \
               + (1/(2 * np.pi * sigma4[0] * sigma4[1])) * \
               np.exp(-0.5 * (((x - mu4[0])**2)/(2 * sigma4[0]**2) + ((y - mu4[1])**2)/(2 * sigma4[1]**2))) \
               + (1/(2 * np.pi * sigma5[0] * sigma5[1])) * \
               np.exp(-0.5 * (((x - mu5[0])**2)/(2 * sigma5[0]**2) + ((y - mu5[1])**2)/(2 * sigma5[1]**2)))
    elif scenario is "three":
        mu, mu2, mu3          = [0,50], [0,0], [0,-50]
        sigma, sigma2, sigma3 = [10,30], [10,30], [10,30]
        func = lambda x,y : (1/(2 * np.pi * sigma[0] * sigma[1])) * \
               np.exp(-0.5 * (((x - mu[0])**2)/(2 * sigma[0]**2) + ((y - mu[1])**2)/(2 * sigma[1]**2))) \
               + (1/(2 * np.pi * sigma2[0] * sigma2[1])) * \
               np.exp(-0.5 * (((x - mu2[0])**2)/(2 * sigma2[0]**2) + ((y - mu2[1])**2)/(2 * sigma2[1]**2))) \
               + (1/(2 * np.pi * sigma3[0] * sigma3[1])) * \
               np.exp(-0.5 * (((x - mu3[0])**2)/(2 * sigma3[0]**2) + ((y - mu3[1])**2)/(2 * sigma3[1]**2)))
    elif scenario is "four":
        if idx < 300:
            mu    = [0, 75]
            sigma = [20, 20]

            func = lambda x,y : (1 / (2 * np.pi * sigma[0] * sigma[1])) * \
                   np.exp(-0.5 * (((x - mu[0])**2)/(2 * sigma[0]**2) + ((y - mu[1])**2)/(2 * sigma[1]**2)))
        elif idx >= 300 and idx < 600:
            mu     = [-40,-40]
            mu2    = [40,-40]
            sigma  = [20,20]
            sigma2 = [20,20]

            func = lambda x,y : (1/(2 * np.pi * sigma[0] * sigma[0])) * \
                   np.exp(-0.5 * (((x - mu[0])**2)/(2 * sigma[0]**2) + ((y - mu[1])**2)/(2 * sigma[1]**2))) \
                   + (1/(2 * np.pi * sigma2[0] * sigma2[1])) * \
                   np.exp(-0.5 * (((x - mu2[0])**2)/(2 * sigma2[0]**2) + ((y - mu2[1])**2)/(2 * sigma2[1]**2)))
        else:
            mu_x    = 0
            mu_y    = 0
            sigma_x = 20
            sigma_y = 20

            func = lambda x,y : (1 / (2 * np.pi * sigma_x * sigma_y)) * \
                   np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2)))

    return func

def create_movie():
    fig = plt.figure(constrained_layout=True, figsize=(1680/192, 1000/192), dpi=192, frameon=True)
    gs  = fig.add_gridspec(2, 2)
    ax  = fig.add_subplot(gs[:,0])
    cax = fig.add_subplot(gs[0,1])
    vax = fig.add_subplot(gs[1,1])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    cax.set_xlim([0, 200])

    vax.set_aspect("equal")
    vax.set_xticks([])
    vax.set_yticks([])
    for spine in vax.spines.values():
        spine.set_visible(False)

    cax.spines['right'].set_visible(False)
    cax.spines['top'].set_visible(False)


    poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
    poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
    poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
    main_poly = Polygon(poly_points)
    transparent_cmap = vorutils.create_transparent_cmap(plt.cm.Blues)

    ext_x, ext_y = main_poly.exterior.xy
    ax.plot(ext_x, ext_y, color='black', linewidth=0.5)
    vax.plot(ext_x, ext_y, color='black', linewidth=0.5)
    # Scenario One.
    path_array = np.load("scenario_one.npy")
    cost_array = np.load("cost_scenario_one.npy")
    vorlist    = np.load("vorinfo_scenario_one.npy", allow_pickle=True)

    initp, = ax.plot(path_array[0,0:20,0], path_array[0,0:20,1], 'r*', markersize=2)
    ax.set_aspect("equal")
    fig.suptitle("Scenario One")
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    joint_xy = choose_function("one")
    x, y = np.mgrid[min_x:max_x, min_y:max_y]
    gauss_val = joint_xy(x, y).ravel()
    cbf = ax.contourf(x, y, gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
    #cb = fig.colorbar(cbf, cax=cbax, orientation="vertical")
    for k in range(1, path_array.shape[0]):
        vlines = list()
        pts, = ax.plot(path_array[k,0:20,0], path_array[k,0:20,1], 'bo', markersize=2, alpha=0.6)
        """
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            ax.plot(*poly.exterior.xy)
        """
        if k == 0:
            cax.plot(k, cost_array[k], color='blue', markersize=1)
        else:
            cax.plot([k, k-1], [cost_array[k], cost_array[k-1]], '-bo', markersize=0.5, linewidth=0.5)

        if k > 0:
            ptsvax, = vax.plot(path_array[k-1,0:20,0], path_array[k-1,0:20,1], 'bo', markersize=2, alpha=0.6)
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            vline, = vax.plot(*poly.exterior.xy, color='black', linewidth=0.5)
            vlines.append(vline)
        plt.show(block=False)
        plt.pause(0.001)
        pts.remove()
        ptsvax.remove()
        [p.remove() for _,p in enumerate(vlines)]
        del pts
        del vline
        del ptsvax
    initp.remove()
    #plt.pause(0.5)

    for i, ob in enumerate(cbf.collections):
        ax.collections.remove(ob)
    cax.cla()
    cax.set_xlim([0, 200])
    # Scenario Two.
    path_array = np.load("scenario_two.npy")
    cost_array = np.load("cost_scenario_two.npy")
    vorlist    = np.load("vorinfo_scenario_two.npy", allow_pickle=True)

    initp, = ax.plot(path_array[0,0:20,0], path_array[0,0:20,1], 'r*', markersize=2)
    ax.set_aspect("equal")
    fig.suptitle("Scenario Two")
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    joint_xy = choose_function("two")
    x, y = np.mgrid[min_x:max_x, min_y:max_y]
    gauss_val = joint_xy(x, y).ravel()
    cbf = ax.contourf(x, y, gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
    #cb = fig.colorbar(cbf, cax=cbax, orientation="vertical")
    for k in range(1, path_array.shape[0]):
        vlines = list()
        pts, = ax.plot(path_array[k,0:20,0], path_array[k,0:20,1], 'bo', markersize=2, alpha=0.6)
        """
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            ax.plot(*poly.exterior.xy)
        """
        if k == 0:
            cax.plot(k, cost_array[k], color='blue', markersize=1)
        else:
            cax.plot([k, k-1], [cost_array[k], cost_array[k-1]], '-bo', markersize=0.5, linewidth=0.5)

        if k > 0:
            ptsvax, = vax.plot(path_array[k-1,0:20,0], path_array[k-1,0:20,1], 'bo', markersize=2, alpha=0.6)
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            vline, = vax.plot(*poly.exterior.xy, color='black', linewidth=0.5)
            vlines.append(vline)
        plt.show(block=False)
        plt.pause(0.001)
        pts.remove()
        ptsvax.remove()
        [p.remove() for _,p in enumerate(vlines)]
        del pts
        del vline
        del ptsvax
    initp.remove()
    #plt.pause(1)

    for i, ob in enumerate(cbf.collections):
        ax.collections.remove(ob)
    cax.cla()
    cax.set_xlim([0, 200])
    # Scenario Three.
    path_array = np.load("scenario_three.npy")
    cost_array = np.load("cost_scenario_three.npy")
    vorlist    = np.load("vorinfo_scenario_three.npy", allow_pickle=True)

    initp, = ax.plot(path_array[0,0:20,0], path_array[0,0:20,1], 'r*', markersize=2)
    ax.set_aspect("equal")
    fig.suptitle("Scenario Three")
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    joint_xy = choose_function("three")
    x, y = np.mgrid[min_x:max_x, min_y:max_y]
    gauss_val = joint_xy(x, y).ravel()
    cbf = ax.contourf(x, y, gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
    #cb = fig.colorbar(cbf, cax=cbax, orientation="vertical")
    for k in range(1, path_array.shape[0]):
        vlines = list()
        pts, = ax.plot(path_array[k,0:20,0], path_array[k,0:20,1], 'bo', markersize=2, alpha=0.6)
        """
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            ax.plot(*poly.exterior.xy)
        """
        if k == 0:
            cax.plot(k, cost_array[k], color='blue', markersize=1)
        else:
            cax.plot([k, k-1], [cost_array[k], cost_array[k-1]], '-bo', markersize=0.5, linewidth=0.5)

        if k > 0:
            ptsvax, = vax.plot(path_array[k-1,0:20,0], path_array[k-1,0:20,1], 'bo', markersize=2, alpha=0.6)
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            vline, = vax.plot(*poly.exterior.xy, color='black', linewidth=0.5)
            vlines.append(vline)
        plt.show(block=False)
        plt.pause(0.001)
        pts.remove()
        ptsvax.remove()
        [p.remove() for _,p in enumerate(vlines)]
        del pts
        del vline
        del ptsvax
    initp.remove()
    #plt.pause(1)

    for i, ob in enumerate(cbf.collections):
        ax.collections.remove(ob)
    cax.cla()
    cax.set_xlim([0, 1000])

    # Scenario Four.
    path_array = np.load("scenario_four.npy")
    cost_array = np.load("cost_scenario_four.npy")
    vorlist    = np.load("vorinfo_scenario_four.npy", allow_pickle=True)

    initp, = ax.plot(path_array[0,0:20,0], path_array[0,0:20,1], 'r*', markersize=2)
    ax.set_aspect("equal")
    fig.suptitle("Scenario Four")
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    joint_xy = choose_function("four")
    x, y = np.mgrid[min_x:max_x, min_y:max_y]
    gauss_val = joint_xy(x, y).ravel()
    cbf = ax.contourf(x, y, gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
    #cb = fig.colorbar(cbf, cax=cbax, orientation="vertical")
    for k in range(1, path_array.shape[0]):
        if k == 300:
            joint_xy = choose_function("four", idx=300)
            x, y = np.mgrid[min_x:max_x, min_y:max_y]
            gauss_val = joint_xy(x, y).ravel()
            for i, ob in enumerate(cbf.collections):
                ax.collections.remove(ob)
            cbf = ax.contourf(x, y, gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
        elif k == 600:
            joint_xy = choose_function("four", idx=600)
            x, y = np.mgrid[min_x:max_x, min_y:max_y]
            gauss_val = joint_xy(x, y).ravel()
            for i, ob in enumerate(cbf.collections):
                ax.collections.remove(ob)
            cbf = ax.contourf(x, y, gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
        vlines = list()
        pts, = ax.plot(path_array[k,0:20,0], path_array[k,0:20,1], 'bo', markersize=2, alpha=0.6)
        """
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            ax.plot(*poly.exterior.xy)
        """
        if k == 0:
            cax.plot(k, cost_array[k], color='blue', markersize=1)
        else:
            cax.plot([k, k-1], [cost_array[k], cost_array[k-1]], '-bo', markersize=0.5, linewidth=0.5)

        if k > 0:
            ptsvax, = vax.plot(path_array[k-1,0:20,0], path_array[k-1,0:20,1], 'bo', markersize=2, alpha=0.6)
        for region in vorlist[k,0]:
            polygon_points = vorlist[k,1][region]
            poly = Polygon(polygon_points).intersection(main_poly)
            polygon = [p for p in poly.exterior.coords]
            #vfill, = ax.fill(*zip(*polygon), alpha=0.5, facecolor='none', edgecolor='black', linewidth=0.5)
            vline, = vax.plot(*poly.exterior.xy, color='black', linewidth=0.5)
            vlines.append(vline)
        plt.show(block=False)
        plt.pause(0.001)
        pts.remove()
        ptsvax.remove()
        [p.remove() for _,p in enumerate(vlines)]
        del pts
        del vline
        del ptsvax
    initp.remove()
    #plt.pause(1)

    for i, ob in enumerate(cbf.collections):
        ax.collections.remove(ob)
    cax.cla()
    cax.set_xlim([0, 100])

    plt.close()

if __name__ == "__main__":
    while True:
        create_movie()

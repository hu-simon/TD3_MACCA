import vorutils
import numpy as np
from shapely.geometry import *
import matplotlib.pyplot as plt

"""
Put in some of the Voronoi lines in the simulation and also
put in the Gaussian plots.
"""

def choose_function(scenario):
    """
    Chooses the lambda function depending on which scenario we are doing.

    Parameters
    ----------
    scenario : String
        String representing the type of scenario we are simulating.

    Returns
    -------
    func : Python lambda function.
        A python lambda function representing the joint density.
    """
    if scenario is "one":
        mu_x = -50
        mu_y = 0
        sigma_x = 20
        sigma_y = 20
        func = lambda x,y : (1/(2 * np.pi * sigma_x * sigma_y)) * \
                   np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2)))
    elif scenario is "two":
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

        func = lambda x,y : (1/(2 * np.pi * sigma_x * sigma_y)) * \
                   np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2))) \
                   + (1/(2 * np.pi * sigma2_x * sigma2_y)) * \
                   np.exp(-0.5 * (((x - mu2_x)**2)/(2 * sigma2_x**2) + ((y - mu2_y)**2)/(2 * sigma2_y**2))) \
                   + (1/(2 * np.pi * sigma3_x * sigma3_y)) * \
                   np.exp(-0.5 * (((x - mu3_x)**2)/(2 * sigma3_x**2) + ((y - mu3_y)**2)/(2 * sigma3_y**2))) \
                   + (1/(2 * np.pi * sigma4_x * sigma4_y)) * \
                   np.exp(-0.5 * (((x - mu4_x)**2)/(2 * sigma4_x**2) + ((y - mu4_y)**2)/(2 * sigma4_y**2))) \
                   + (1/(2 * np.pi * sigma5_x * sigma5_y)) * \
                   np.exp(-0.5 * (((x - mu5_x)**2)/(2 * sigma5_x**2) + ((y - mu5_y)**2)/(2 * sigma5_y**2)))
    elif scenario is "three":
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

        func = lambda x,y : (1/(2 * np.pi * sigma_x * sigma_y)) * \
                   np.exp(-0.5 * (((x - mu_x)**2)/(2 * sigma_x**2) + ((y - mu_y)**2)/(2 * sigma_y**2))) \
                   + (1/(2 * np.pi * sigma2_x * sigma2_y)) * \
                   np.exp(-0.5 * (((x - mu2_x)**2)/(2 * sigma2_x**2) + ((y - mu2_y)**2)/(2 * sigma2_y**2))) \
                   + (1/(2 * np.pi * sigma3_x * sigma3_y)) * \
                   np.exp(-0.5 * (((x - mu3_x)**2)/(2 * sigma3_x**2) + ((y - mu3_y)**2)/(2 * sigma3_y**2)))

    return func

if __name__ == "__main__":
    while True:
        fig, ax = plt.subplots(1,1)

        poly_points = [[0,100], [-95,31], [-59,-81], [59,-81], [95,31]]
        poly_points_x = np.array([poly_points[i][0] for i in range(len(poly_points))])
        poly_points_y = np.array([poly_points[i][1] for i in range(len(poly_points))])
        main_poly = Polygon(poly_points)
        ext_x, ext_y = main_poly.exterior.xy
        plt.plot(ext_x, ext_y, color='black', linewidth=0.7)
        min_x, max_x, min_y, max_y = plt.axis()
        transparent_cmap = vorutils.create_transparent_cmap(plt.cm.Blues)
        x, y = np.mgrid[min_x:max_x, min_y:max_y]

        # Scenario One
        path_array = np.load('scenario_one.npy')
        #plt.clf()
        plt.title('Scenario One')
        plt.plot(path_array[0,0:20,0], path_array[0,0:20,1], 'g*', markersize=3)
        joint_xy = choose_function("one")
        gauss_val = joint_xy(x,y).ravel()
        cbf = ax.contourf(x,y,gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
        cb = plt.colorbar(cbf, use_gridspec=True)
        for k in range(1, path_array.shape[0]):
            plt.axis('equal')
            plt.plot(ext_x, ext_y, color='black', linewidth=0.7)
            pts, = plt.plot(path_array[k,0:20,0], path_array[k,0:20,1], 'ro', markersize=3, alpha=0.8)
            plt.axis('equal')
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.title('Scenario One')
            plt.show(block=False)
            plt.pause(0.0001)
            pts.remove()

        # Scenario Two
        cb.remove()
        for i, ob in enumerate(cbf.collections):
            plt.gca().collections.remove(ob)
        path_array = np.load('scenario_two.npy')
        plt.title('Scenario Two')
        plt.plot(ext_x, ext_y, color='black', linewidth=0.7)
        min_x, max_x, min_y, max_y = plt.axis()
        plt.plot(path_array[0,0:20,0], path_array[0,0:20,1], 'g*', markersize=3)
        joint_xy = choose_function("two")
        gauss_val = joint_xy(x,y).ravel()
        cbf = ax.contourf(x,y,gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
        cb = plt.colorbar(cbf, use_gridspec=True)
        for k in range(1, path_array.shape[0]):
            plt.axis('equal')
            plt.plot(ext_x, ext_y, color='black', linewidth=0.7)
            pts, = plt.plot(path_array[k,0:20,0], path_array[k,0:20,1], 'ro', markersize=3, alpha=0.8)
            plt.axis('equal')
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.title('Scenario Two')
            plt.show(block=False)
            plt.pause(0.0001)
            pts.remove()

        # Scenario Three
        cb.remove()
        for i, ob in enumerate(cbf.collections):
            plt.gca().collections.remove(ob)
        path_array = np.load('scenario_three.npy')
        plt.title('Scenario Three')
        plt.plot(path_array[0,0:20,0], path_array[0,0:20,1], 'g*', markersize=3)
        joint_xy = choose_function("three")
        gauss_val = joint_xy(x,y).ravel()
        cb = ax.contourf(x,y,gauss_val.reshape(x.shape[0], y.shape[1]), 15, cmap=transparent_cmap)
        plt.colorbar(cb)
        for k in range(1, path_array.shape[0]):
            plt.axis('equal')
            plt.plot(ext_x, ext_y, color='black', linewidth=0.7)
            pts, = plt.plot(path_array[k,0:20,0], path_array[k,0:20,1], 'ro', markersize=3, alpha=0.8)
            plt.axis('equal')
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.title('Scenario Three')
            plt.show(block=False)
            plt.pause(0.01)
            pts.remove()

        plt.close()

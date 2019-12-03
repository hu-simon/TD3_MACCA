import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import *
from research_utils.vorutils import helpers
from scipy.spatial import *

points = helpers.generate_random_points(0, 10, 5)
points_x = np.array([points[i][0] for i in range(len(points))])
points_y = np.array([points[i][1] for i in range(len(points))])

vor = Voronoi(points) 
regions, vertices = helpers.create_finite_voronoi_2d(vor)

min_x = vor.min_bound[0] - 1
max_x = vor.max_bound[0] + 1
min_y = vor.min_bound[1] - 1
max_y = vor.max_bound[1] + 1
bounding_box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

for region in regions:
    polygon_points = vertices[region]
    poly = Polygon(polygon_points).intersection(bounding_box)
    polygon = [p for p in poly.exterior.coords]
    print(polygon)
    plt.fill(*zip(*polygon), alpha=0.8, facecolor='none', edgecolor='black', linewidth=1)

plt.plot(points_x, points_y, 'ko', markersize=3)
plt.axis('equal')
plt.show()

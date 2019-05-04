#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

# Import functions to plot figures
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    neighborhoods = []

    for query in queries:
        indices = np.where(np.linalg.norm(supports - query, axis = 1) < radius)[0]
        neighborhoods.append(supports[indices])

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    neighborhoods = []

    for query in queries:
        indices = np.argsort(np.linalg.norm(supports - query))[:k]
        neighborhoods.append(supports[indices])

    return neighborhoods


class neighborhood_grid:

    def __init__(self, points, voxel_size):

        #
        #   Tips :
        #       > "__init__" method is called when you create an object of this class with the line :
        #         grid = neighborhood_grid(points, voxel_size)
        #       > You need to keep here the variables that you want to use later (in the query_radius method).
        #         Just call them "self.XXX". The "__init__" method does not need to return anything
        #

        # Example : save voxel size for later use
        self.grid_voxel_size = voxel_size
        self.points = points
        self.initialize_grid()


    def initialize_grid(self):

        min_coords, max_coords = self.points.min(0), self.points.max(0)
        pts_indices = np.floor((self.points - min_coords) / self.grid_voxel_size)

        coord_grids, pts_grids = np.unique(pts_indices, return_inverse = True, axis = 0)
        num_grids = pts_grids.max() + 1 

        self.grids = {}

        for i in range(num_grids):
            self.grids[tuple(coord_grids[i].astype(int))] = np.where(pts_grids == i)[0]

        self.min_coords = min_coords


    def query_radius(self, queries, radius):

        #
        #   Tips :
        #       > To speed up the query, you need to find for each point, the part of the grid where its
        #         neighbors can be.
        #       > Then loop over the cells in this part of the grid to find the real neighbors
        #

        neighborhoods = []
        offsets = int(np.ceil(radius / self.grid_voxel_size)) 

        for query in queries:
            query_grid = np.floor((query - self.min_coords) / self.grid_voxel_size).astype(int)
            potential_pts = []
            for i in range(query_grid[0] - offsets, query_grid[0] + offsets + 1):
                for j in range(query_grid[1] - offsets, query_grid[1] + offsets + 1):
                    for k in range(query_grid[2] - offsets, query_grid[2] + offsets + 1):
                        test_voxel = tuple([i, j, k])
                        if test_voxel in self.grids:
                            potential_pts.append(self.grids[tuple(test_voxel)])

            potential_pts = self.points[np.unique(np.hstack(potential_pts))]
            indices = np.where(np.linalg.norm(potential_pts - query, axis = 1) < radius)[0]
            neighborhoods.append(potential_pts[indices])

        return neighborhoods


class hierarchical_search:

    def __init__(self, points, leaf_size):

        self.tree = KDTree(points, leaf_size = leaf_size)
        self.points = points

    def query_radius(self, queries, radius):

        indices = self.tree.query_radius(queries, radius)
        neighborhoods = [self.points[indice] for indice in indices]

        return neighborhoods


def plot_figure(x, y, x_label, y_label, title, legend = None):

    plt.figure(figsize = (6, 6))
    if legend is None:
        plt.plot(x, y)
    else:
        for i in range(len(y)):
            plt.plot(x, y[i], label = legend[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.title(title)
    if legend is not None:
        plt.legend()
    plt.show()

# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

    # Grid neighborhood verification
    # ******************************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 10
        radius = 0.2
        voxel_size = 0.2

        # Create grid structure
        grid = neighborhood_grid(points, voxel_size)

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Get neighborhoods with the grid
        grid_neighborhoods = grid.query_radius(queries, radius)

        # Get neighborhoods with brute force
        brute_neighborhoods = brute_force_spherical(queries, points, radius)


        # Compare all neighborhoods
        print('\nVerification of grid neighborhoods :')
        for n1, n2 in zip(grid_neighborhoods, brute_neighborhoods):
            if n1.shape[0] != n2.shape[0]:
                print('ERROR FOUND : wrong amount of neighbors')
            else:
                diffs = np.unique(n1, axis = 0) - np.unique(n2, axis = 0)
                error = np.sum(np.abs(diffs))
                if error > 0:
                    print('ERROR FOUND : wrong neighbors')
                else:
                    print('This neighborhood is good')

    # Grid neighborhood timings
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 10
        radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        query_times = [[], [], []]
        voxel_sizes = [0.2, 0.5, 1]

        # Create grid structure
        grids = [neighborhood_grid(points, voxel_size) for voxel_size in voxel_sizes]

        print("Begin queries...")

        for radius in radius_values:

            print("Radius: %f" % radius)
            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            # Search spherical
            for i in range(3):
                t0 = time.time()
                neighborhoods = grids[i].query_radius(queries, radius)
                t1 = time.time()        
                query_times[i].append((t1 - t0) / num_queries)


        plot_figure(radius_values, query_times, "radius", "s", 
                    "Timing to query 1 point W.R.T. Search Radius",
                    ["l = 0.2m", "l = 0.5m", "l = 1m"])
        

    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 1000
        #radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        radius = 0.2
        leaf_sizes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 80, 100]
        query_times = []

        for leaf_size in leaf_sizes:

            # Create grid structure
            kdtree = hierarchical_search(points, leaf_size = 10)

            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            # Search spherical
            t0 = time.time()
            neighborhoods = kdtree.query_radius(queries, radius)
            t1 = time.time()        
            print('{:d} spherical neighborhood computed in {:.3f} seconds'.format(num_queries, t1 - t0))

            query_times.append(t1 - t0)

        plot_figure(leaf_sizes, query_times, "leaf size", 's', 
                    'Timing to query 1 point W.R.T. Leaf size')


    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000
        radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        query_times = []

        for radius in radius_values:

            # Create grid structure
            kdtree = hierarchical_search(points, leaf_size = 10)

            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            # Search spherical
            t0 = time.time()
            neighborhoods = kdtree.query_radius(queries, radius)
            t1 = time.time()        
            print('{:d} spherical neighborhood computed in {:.3f} seconds'.format(num_queries, t1 - t0))

            query_times.append((t1 - t0) / num_queries)

        plot_figure(radius_values, query_times, "radius", "s",
                    'Timing to query 1 point W.R.T. search radius')
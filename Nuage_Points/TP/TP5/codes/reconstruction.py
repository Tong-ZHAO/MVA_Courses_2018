#
#
#      0===========================================================0
#      |              TP5 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 02/02/2018
#



# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply, write_obj

# Import time package
import time

from skimage import measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Hoppe surface reconstruction
def compute_hoppe(points, normals, volume, number_cells, min_grid, length_cell):
    """Compute the hoppe implicit function on a regular grid

    Params:
        points       (np.array): of size N x 3
        normals      (np.array): of size N x 3
        volume       (np.array): of size (num_cells + 1)^3, zero by default
        number_cells (float)   : number of cells on each dimension
        min_grid     (np.array): the left-top corner of the grid
        length_cell  (float)   : the length of each voxel

    Returns:
        The result is in the variable 'volume', whose size is (num_cells + 1)^3 
    """

    my_tree = KDTree(points, 10)

    xgrid, ygrid, zgrid = np.mgrid[0 : (number_cells + 1), 0 : (number_cells + 1), 0 : (number_cells + 1)]
    grid_idx = np.vstack((xgrid.reshape((-1)), ygrid.reshape((-1)), zgrid.reshape((-1)))).T
    grid_pts = grid_idx * length_cell + min_grid.reshape((1, -1))

    nbs_list = np.squeeze(my_tree.query(grid_pts, k = 1, return_distance = False))

    values = (normals[nbs_list] * (grid_pts - points[nbs_list])).sum(1)
    volume[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]] = values

				
# EIMLS surface reconstruction
def compute_eimls(points, normals, volume, number_cells, min_grid, length_cell, h = 0.003, number_nb = 10):
    """Compute the hoppe implicit function on a regular grid

    Params:
        points       (np.array): of size N x 3
        normals      (np.array): of size N x 3
        volume       (np.array): of size (num_cells + 1)^3, zero by default
        number_cells (float)   : number of cells on each dimension
        min_grid     (np.array): the left-top corner of the grid
        length_cell  (float)   : the length of each voxel
        h            (float)   : minimum h
        nuimber_nb   (int)     : number of nearest neighbors

    Returns:
        The result is in the variable 'volume', whose size is (num_cells + 1)^3 
    """

    
    my_tree = KDTree(points, 10)

    xgrid, ygrid, zgrid = np.mgrid[0 : (number_cells + 1), 0 : (number_cells + 1), 0 : (number_cells + 1)]
    grid_idx = np.vstack((xgrid.reshape((-1)), ygrid.reshape((-1)), zgrid.reshape((-1)))).T
    grid_pts = grid_idx * length_cell + min_grid.reshape((1, -1))

    nbs_dist, nbs_list = my_tree.query(grid_pts, k = number_nb, return_distance = True)
    nbs_h = np.clip(nbs_dist / 4., h, None)
    nbs_coeff = np.exp(-np.power(nbs_dist / nbs_h, 2))

    nbs_value = np.zeros_like(nbs_dist)

    for i in range(number_nb):
        nbs_value[:, i] = (normals[nbs_list[:, i]] * (grid_pts - points[nbs_list[:, i]])).sum(1)

    volume[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]] = (nbs_value * nbs_coeff).sum(1) / nbs_coeff.sum(1)
				
				
if __name__ == '__main__':

    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = points.min(0)
    max_grid = points.max(0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10 * (max_grid - min_grid)
    max_grid = max_grid + 0.10 * (max_grid - min_grid)

	# Number_cells is the number of voxels in the grid in x, y, z axis
    number_cells = 200 #100
    length_cell = (max_grid - min_grid) / number_cells
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    volume = np.zeros((number_cells + 1,number_cells + 1,number_cells + 1), dtype = np.float32)

	# Compute the scalar field in the grid
    #compute_hoppe(points, normals, volume, number_cells, min_grid, length_cell)
    compute_eimls(points, normals, volume, number_cells, min_grid, length_cell)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes_lewiner(volume, level=0.0, spacing=(length_cell[0],length_cell[1],length_cell[2]))

    write_obj(verts, faces, "../data/bunny_recst_eilms.obj")
	
	# Plot the mesh using matplotlib 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, number_cells*length_cell[0])
    ax.set_ylim(0, number_cells*length_cell[1])
    ax.set_zlim(0, number_cells*length_cell[2])
    plt.axis('off')
    plt.show()
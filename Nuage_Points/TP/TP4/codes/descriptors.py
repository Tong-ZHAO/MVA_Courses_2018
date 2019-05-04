#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def local_PCA(points):
    """Calculate PCA on a set of points

    Params:
        points(np.array): point set of size n x d

    Returns:
        eigenvalues (np.array): of size (d,), the eigenvalues in descending order
        eigenvectors(np.array): of size (d, d), each column is an eigenvector
    """

    centroid = points.mean(axis = 0, keepdims = True)
    cpoints = points - centroid

    cov_mat = cpoints.T.dot(cpoints) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    eigenvalues = eigenvalues[::-1]
    eigenvectors = np.flip(eigenvectors, 1)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius):
    """Compute PCA on the neighborhoods of all query_points in cloud_pointos

    Params:
        query_points(np.array): of size N1 x d, the points to be queried
        cloud_points(np.array): of size N2 x d, the whole point cloud
        radius      (float)   : the param to choose neighbors

    Returns:
        all_eigenvalues (np.array): of size N1 x d
        all_eigenvectors(np.array): of size N2 x d x d
    """

    my_tree = KDTree(cloud_points, 10)
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    nbs_list = my_tree.query_radius(query_points, radius)

    for i, nbs in enumerate(nbs_list):
        my_nbs = cloud_points[nbs]
        my_eigenvalues, my_eigenvectors = local_PCA(my_nbs)

        all_eigenvalues[i] = my_eigenvalues
        all_eigenvectors[i] = my_eigenvectors 
    
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    """Compute the features for all query points in the cloud

    Params:
        query_points(np.array): of size N1 x d, the points to be queried
        cloud_points(np.array): of size N2 x d, the whole point cloud
        radius      (float)   : the param to choose neighbors

    Returns:
        verticality (np.array): the angle between the normals and the vertical direction
        linearity   (np.array): the local linearity
        planarity   (np.array): the local planarity
        sphericity  (np.array): the local sphericity
    """

    all_eigenvalues, all_eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)
    all_normals = all_eigenvectors[:, :, -1]
    ez = np.array([0, 0, 1]).reshape((1, -1))

    verticality = 2 * np.arcsin(np.abs(all_normals.dot(ez.T))) / np.pi
    linearity = 1 - all_eigenvalues[:, 1] / np.clip(all_eigenvalues[:, 0], 1e-6, None)
    planarity = (all_eigenvalues[:, 1] - all_eigenvalues[:, 2]) / np.clip(all_eigenvalues[:, 0], 1e-6, None)
    sphericity = all_eigenvalues[:, 2] / np.clip(all_eigenvalues[:, 0], 1e-6, None)

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = local_PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        cloud_eigenvalues, cloud_eigenvectors = neighborhood_PCA(cloud, cloud, 1.) 
        cloud_normal = cloud_eigenvectors[:, :, -1]

        write_ply('Lille_small_normal.ply', [cloud, cloud_normal], ['x', 'y', 'z', 'nx', 'ny', 'nz'])


    # Features computation
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        features = compute_features(cloud, cloud, 1.)

        write_ply('Lille_small_feature.ply', [cloud] + list(features), ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])

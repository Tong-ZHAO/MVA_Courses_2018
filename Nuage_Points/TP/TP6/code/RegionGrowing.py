#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Plane detection by region growing
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
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


def compute_planarities_and_normals(points, radius):
    """Compute planarities and normals for a point cloud

    Params:
        points (np.array): of size N x 3
        radius (float)   : the radius to choose neighbors

    Returns:
        planarities (np.array) of size N, (lambda_2 - lambda_3) / lambda_1
        normals     (np.array) of size N x 3, v_3
    """

    all_eigenvalues, all_eigenvectors = neighborhood_PCA(points, points, radius)

    normals = all_eigenvectors[:, :, -1]
    planarities = (all_eigenvalues[:, 1] - all_eigenvalues[:, 2]) / np.clip(all_eigenvalues[:, 0], 1e-6, None)

    return planarities, normals


def region_criterion(p1, p2, n1, n2, thresh1 = 0.1, thresh2 = 10):
    """The criterion for adding a point in the plane

    Our criterion is based on the distance from the point to the plane 
    and the angle between two normals.

    Params:
        p1   (np.array): of size 3, the reference point
        p2   (np.array): of size 3, the candidate point
        n1   (np.array): of size 3, the reference normal
        n2   (np.array): of size 3, the candidate normal
        thresh1 (float): the distance should below this threshold
        thresh2 (float): the angle should below this threshold

    Returns:
        flag (bool): True if the candidate meets our criterion, False otherwise.
    """

    dist_plane = np.abs(n1.dot(p2- p1))
    angle = 180 * np.arccos(np.abs(np.clip(n1.dot(n2), -1., 1.))) / np.pi

    return True if dist_plane < thresh1 and angle < thresh2 else False


def queue_criterion(p, thresh = 0.85):
    """The criteria for adding a point to the queue

    Params:
        p   (np.array): of size 3, the candidate point
        thresh (float): the planarity should be larger than this threshold

    Returns:
        flag (bool): True if the candidate meets our criteria, False otherwise. 
    """

    return True if p > thresh else False


def RegionGrowing(cloud, normals, planarities, radius):
    """Find one plane from the point cloud by using Region Growing algorithm.

    Params:
        cloud       (np.array): of size N x 3, the point cloud
        normals     (np.array): of size N x 3, the normals of the point cloud
        planarities (np.array): of size N, the planarities of the point cloud
        radius         (float): the radius to find the neighbors of a point

    Returns:
        region (np.array): of size N, bool flags to indicate if one point belows to the plane or not
    """

    N = len(cloud)
    region = np.zeros(N, dtype=bool)

    # Find the seed point
    queue = [np.argmax(planarities)]
    region[queue[0]] = True

    # Find neighbors for all points
    my_tree = KDTree(cloud, 10)
    all_neighbors = my_tree.query_radius(cloud, radius)

    while len(queue) > 0:
        # Get reference point
        point_ind = queue.pop(0)
        nbs_ind = [ind for ind in all_neighbors[point_ind] if region[ind] == False]
        # Check neighbors
        for nb_ind in nbs_ind:
            flag_region = region_criterion(cloud[point_ind], cloud[nb_ind], normals[point_ind], normals[nb_ind])
            # Add to plane
            if flag_region:
                region[nb_ind] = True
                # Add to queue
                if queue_criterion(planarities[nb_ind]):
                    queue.append(nb_ind)

    return region


def multi_RegionGrowing(cloud, normals, planarities, radius, NB_PLANES = 2):
    """Find N planes from the point cloud by using Region Growing algorithm.

    Params:
        cloud       (np.array): of size N x 3, the point cloud
        normals     (np.array): of size N x 3, the normals of the point cloud
        planarities (np.array): of size N, the planarities of the point cloud
        radius         (float): the radius to find the neighbors of a point
        NB_PLANES        (int): the number of planes to extract

    Returns:
        plane_inds     (np.array): of size N, bool flags to indicate if a point belongs to one plane or not
        remaining_inds (np.array): of size N, indices to indicate which points are left
        plane_labels   (np.array): of size N, integers to indicate the label of the plane which the point belongs to 
                                   (0 if the point doesn't belong to any plane)
    """

    N = len(cloud)
    plane_inds = np.zeros((N,)).astype(bool)
    plane_labels = np.zeros((N,)).astype(int)
    remaining_inds = np.arange(N).astype(int)

    for i in range(NB_PLANES):
        best_region = RegionGrowing( cloud[remaining_inds], 
                                     normals[remaining_inds], 
                                     planarities[remaining_inds],
                                     radius )
        plane_inds[remaining_inds[best_region]] = True
        plane_labels[remaining_inds[best_region]] = i + 1
        remaining_inds = remaining_inds[best_region == False]

    return plane_inds, remaining_inds, plane_labels


# ----------------------------------------------------------------------------------------------------------------------
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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    N = len(points)

    # Computes normals of the whole cloud
    # ***********************************
    #

    # Parameters for normals computation
    radius = 0.2

    # Computes normals of the whole cloud
    t0 = time.time()
    planarities, normals = compute_planarities_and_normals(points, radius)
    t1 = time.time()
    print('normals and planarities computation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../planarities.ply',
              [points, planarities],
              ['x', 'y', 'z', 'planarities'])

    # Find a plane by Region Growing
    # ******************************
    #

    if False:
        # Define parameters of Region Growing
        radius = 0.2

        # Find a plane by Region Growing
        t0 = time.time()
        region = RegionGrowing(points, normals, planarities, radius)
        t1 = time.time()
        print('Region Growing done in {:.3f} seconds'.format(t1 - t0))

        # Get inds from bollean array
        plane_inds = region.nonzero()[0]
        remaining_inds = (1 - region).nonzero()[0]

        # Save the best plane
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds], planarities[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'planarities'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds], planarities[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'planarities'])

    # Find multiple in the cloud
    # ******************************
    #

    if True:
        # Define parameters of multi_RANSAC
        radius = 0.1
        NB_PLANES = 10

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RegionGrowing(points, normals, planarities, radius, NB_PLANES)
        t1 = time.time()
        print('multi RegionGrowing done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels[plane_inds].astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')

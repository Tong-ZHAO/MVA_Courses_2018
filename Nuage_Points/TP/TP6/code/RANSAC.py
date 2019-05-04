#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
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


def compute_plane(points):
    """Compute a plane from the given point cloud

    Params:
        points (np.array): of size 3 x 3, the points on the plane

    Returns:
        point  (np.array): of size 3 x 1, one reference point on the plane
        normal (np.array): of size 3 x 1, the normal of the plane
    """

    point = points[0].reshape((3, 1))
    normal = np.cross(points[1] - point.T, points[2] - point.T).reshape((3, 1))
    
    return point, normal / np.linalg.norm(normal)


def in_plane(points, ref_pt, normal, threshold_in = 0.1):
    """Find all points on the given plane from a point cloud

    Params:
        points (np.array): of size n x 3, the point cloud
        ref_pt (np.array): of size 3 x 1, the reference point of the plane
        normal (np.array): of size 3 x 1, the normal of the plane
        threshold (float): the threshold of the distance between a point and the plane

    Returns:
        indices (np.array): of size n x 3, the bool flags which indicates if a point is in the plane or not
    """

    dists = np.abs((points - ref_pt.T).dot(normal))
    indices = dists < threshold_in
    
    return np.squeeze(indices)


def RANSAC(points, NB_RANDOM_DRAWS = 100, threshold_in = 0.1):
    """Find one plane from a point cloud by RANSAC algorithm

    Params:
        points       (np.array): of size N x 3, the whole point cloud
        NB_RANDOM_DRAWS (float): the number of draws each time we choose a plane
        threshold_in    (float): the tolerable distance from a point to the plane

    Returns:
        best_ref_pt (np.array): of size 3 x 1, the reference point of the plane
        best_normal (np.array): of size 3 x 1, the reference normal of the plane
    """
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    best_score = 0
    N = len(points)
    
    for _ in range(NB_RANDOM_DRAWS):
        sample_points = points[np.random.choice(N, 3, replace = False)]
        sample_ref, sample_normal = compute_plane(sample_points)
        sample_indices = in_plane(points, sample_ref, sample_normal, threshold_in)

        if sample_indices.sum() > best_score:
            best_score = sample_indices.sum()
            best_ref_pt = sample_ref
            best_normal = sample_normal 
                
    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS = 100, threshold_in = 0.1, NB_PLANES = 2):
    """Find m plane from a point cloud by RANSAC algorithm

    Params:
        points       (np.array): of size N x 3, the whole point cloud
        NB_RANDOM_DRAWS (float): the number of draws each time we choose a plane
        threshold_in    (float): the tolerable distance from a point to the plane
        NB_PLANES         (int): the number of planes we extracted

    Returns:
        plane_inds     (np.array): of size N, bool flags to indicate if a point belongs to one plane or not
        remaining_inds (np.array): of size N, indices to indicate which points are left
        plane_labels   (np.array): of size N, integers to indicate the label of the plane which the point belongs to 
                                   (0 if the point doesn't belong to any plane)
    """
    
    N = len(points)
    plane_inds = np.zeros((N,)).astype(bool)
    plane_labels = np.zeros((N,)).astype(int)
    remaining_inds = np.arange(N).astype(int)

    for i in range(NB_PLANES):
        best_ref_pt, best_normal = RANSAC(points[remaining_inds], NB_RANDOM_DRAWS, threshold_in)
        curr_inds = in_plane(points[remaining_inds], best_ref_pt, best_normal, threshold_in)
        plane_inds[remaining_inds[curr_inds]] = True
        plane_labels[remaining_inds[curr_inds]] = i + 1
        remaining_inds = remaining_inds[curr_inds == False]

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

    print(points.shape)

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if False:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]

        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if False:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]
        print("Find %d points on the plane." % points_in_plane.sum())

        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:

        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 200
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels[plane_inds].astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')

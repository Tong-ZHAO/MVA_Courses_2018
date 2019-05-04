#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from utils.visu import show_ICP


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    pm_data, pm_ref = data.mean(1).reshape((-1, 1)), ref.mean(1).reshape((-1, 1))
    q_data, q_ref = data - pm_data, ref - pm_ref
    H = q_data.dot(q_ref.T)
    U, S, VT = np.linalg.svd(H)

    while True:
        R = (U.dot(VT)).T
        T = pm_ref - R.dot(pm_data)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
        else:
            return R, T


def RMS(pts_1, pts_2):
    """
    Inputs:
        pts_1 = (d x N_data)
        pts_2 = (d x N_data)
    """

    total_sqrms = np.power(np.linalg.norm(pts_1 - pts_2, axis = 0), 2)
    rms = np.sqrt(total_sqrms.sum() / pts_1.shape[1])

    return rms


def plot_rms(rms_list, title):

    plt.figure()
    plt.plot(rms_list)
    plt.xlabel("iter")
    plt.ylabel("rms")
    plt.title(title)
    plt.grid()
    plt.show()


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (N_data,) array of indices. This is the list of those
        arrays at each iteration
        rms_list = list of the RMS error at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = []

    ref_tree = KDTree(ref.T, leaf_size = 2) 
    rms, iter = np.inf, 0

    while iter < max_iter and rms > RMS_threshold:
        matching_idx = ref_tree.query(data_aligned.T, k = 1, return_distance = False)
        matching_pts = ref[:, np.squeeze(matching_idx)]
        R, T = best_rigid_transform(data_aligned, matching_pts)

        if iter == 0:
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R.dot(R_list[-1]))
            T_list.append(R.dot(T_list[-1]) + T)

        neighbors_list.append(matching_idx.reshape((-1)))
        data_aligned = R.dot(data_aligned) + T
        rms = RMS(data_aligned, matching_pts)
        rms_list.append(rms)
        iter += 1

    return data_aligned, R_list, T_list, neighbors_list, rms_list


def icp_point_to_point_stochastic(data, ref, max_iter, RMS_threshold, sampling_limit, final_overlap = 1.):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        sampling_limit = the number of chosen points at each iteration
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (N_data,) array of indices. This is the list of those
        arrays at each iteration
        rms_list = list of the RMS error at each iteration
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = []

    ref_tree = KDTree(ref.T, leaf_size = 2) 
    rms, iter, num_pts = np.inf, 0, data_aligned.shape[1]

    while iter < max_iter and rms > RMS_threshold:
        idx = np.random.choice(num_pts, sampling_limit, replace = False)
        data_aligned_sub = data_aligned[:, idx]
        matching_dist, matching_idx = ref_tree.query(data_aligned_sub.T, k = 1)

        if final_overlap < 1:
            keep_num = int(final_overlap * data_aligned_sub.shape[1])
            keep_idx = np.argsort(np.squeeze(matching_dist))[:keep_num]
            matching_idx = matching_idx[keep_idx]
            data_aligned_sub = data_aligned_sub[:, keep_idx] 

        matching_pts = ref[:, np.squeeze(matching_idx)]
        R, T = best_rigid_transform(data_aligned_sub, matching_pts)

        if iter == 0:
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R.dot(R_list[-1]))
            T_list.append(R.dot(T_list[-1]) + T)

        neighbors_list.append(matching_idx.reshape((-1)))
        data_aligned = R.dot(data_aligned) + T
        rms = RMS(data_aligned_sub, matching_pts)
        rms_list.append(rms)
        iter += 1

    return data_aligned, R_list, T_list, neighbors_list, rms_list

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

        # Load clouds
        bunny_o_file = read_ply(bunny_o_path)
        bunny_r_file = read_ply(bunny_r_path)

        bunny_o_pts = np.vstack((bunny_o_file['x'], bunny_o_file['y'], bunny_o_file['z'])) # data
        bunny_r_pts = np.vstack((bunny_r_file['x'], bunny_r_file['y'], bunny_r_file['z'])) # ref

        # Find the best transformation
        R, T = best_rigid_transform(bunny_o_pts, bunny_r_pts)

        # Apply the tranformation
        bunny_m_pts = R.dot(bunny_o_pts) + T

        # Save cloud
        write_ply('../bunny_transform.ply', [bunny_m_pts.T], ['x', 'y', 'z'])

        # Compute RMS
        rms = RMS(bunny_m_pts, bunny_r_pts)

        # Print RMS
        print("Final RMS: %f" % rms)
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds
        ref2D_file = read_ply(ref2D_path)
        data2D_file = read_ply(data2D_path)

        ref2D_pts = np.vstack((ref2D_file['x'], ref2D_file['y'])) # data
        data2D_pts = np.vstack((data2D_file['x'], data2D_file['y'])) # ref

        # Apply ICP
        max_iter = 30
        RMS_threshold = 0.06
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(data2D_pts, ref2D_pts, max_iter, RMS_threshold)

        # Show ICP
        #write_ply('../data2D_transform.ply', [data_aligned.T], ['x', 'y']) 
        show_ICP(data2D_pts, ref2D_pts, R_list, T_list, neighbors_list)
        plot_rms(rms_list, "RMS on Data2D")

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'

        # Load clouds
        bunny_o_file = read_ply(bunny_o_path)
        bunny_p_file = read_ply(bunny_p_path)

        bunny_o_pts = np.vstack((bunny_o_file['x'], bunny_o_file['y'], bunny_o_file['z'])) # data
        bunny_p_pts = np.vstack((bunny_p_file['x'], bunny_p_file['y'], bunny_p_file['z'])) # ref 

        # Apply ICP
        max_iter = 30
        RMS_threshold = 1e-4
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(bunny_p_pts, bunny_o_pts, max_iter, RMS_threshold)

        # Show ICP
        #write_ply('../bunny_transform.ply', [data_aligned.T], ['x', 'y', 'z']) 
        show_ICP(bunny_p_pts, bunny_o_pts, R_list, T_list, neighbors_list)
        plot_rms(rms_list, "RMS on Bunny")



    # Fast ICP
    # ********
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        NDDC_1_file= read_ply(NDDC_1_path)
        NDDC_2_file = read_ply(NDDC_2_path)

        NDDC_1_pts = np.vstack((NDDC_1_file['x'], NDDC_1_file['y'], NDDC_1_file['z'])) # data
        NDDC_2_pts = np.vstack((NDDC_2_file['x'], NDDC_2_file['y'], NDDC_2_file['z'])) # ref 

        # Apply fast ICP for different values of the sampling_limit parameter
        max_iter = 100
        RMS_threshold = 1e-4
        sampling_limit = 50000
        final_overlap = 0.7
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point_stochastic(NDDC_2_pts, NDDC_1_pts, max_iter, RMS_threshold, sampling_limit, final_overlap)

        # Plot RMS
        #
        # => To plot something in python use the function plt.plot() to create the figure and 
        #    then plt.show() to display it
        write_ply('../data/NDDC_transform.ply', [data_aligned.T], ['x', 'y', 'z']) 
        plot_rms(rms_list, "RMS on NDDC (num = %d)" % sampling_limit)

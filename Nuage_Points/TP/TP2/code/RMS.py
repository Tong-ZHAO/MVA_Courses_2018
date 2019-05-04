import numpy as np
from utils.ply import write_ply, read_ply
from sklearn.neighbors import KDTree


def RMS(pts_1, pts_2, overlap = 1.):
    """
    Inputs:
        pts_1 = (d x N_data)
        pts_2 = (d x N_data)
        overlap = a float between 0 and 1
    """

    total_sqrms = np.power(np.linalg.norm(pts_1 - pts_2, axis = 0), 2)
    if overlap < 1.:
        keep_num = int(pts_1.shape[1] * overlap)
        total_sqrms = np.sort(total_sqrms)[:keep_num]

    rms = np.sqrt(total_sqrms.sum() / len(total_sqrms))

    return rms


def read_pts(file_path, dim = 3):

    my_file = read_ply(file_path)
    my_pts = np.vstack((my_file['x'], my_file['y'], my_file['z'])) if dim == 3 else np.vstack((my_file['x'], my_file['y']))

    return my_pts


if __name__ == '__main__':

    file_path_ref = "../data/Notre_Dame_Des_Champs_1.ply"
    file_path_data = "../data/NDDC_transform.ply"

    pts_ref = read_pts(file_path_ref)
    pts_data = read_pts(file_path_data)

    ref_tree = KDTree(pts_ref.T, leaf_size = 10)
    idx = np.squeeze(ref_tree.query(pts_data.T, k = 1, return_distance = False))

    print("RMS on whole point cloud: %f" % RMS(pts_data, pts_ref[:, idx]))
    print("RMS on 90%% point cloud: %f" % RMS(pts_data, pts_ref[:, idx], 0.9))
    print("RMS on 70%% point cloud: %f" % RMS(pts_data, pts_ref[:, idx], 0.7))
    print("RMS on 50%% point cloud: %f" % RMS(pts_data, pts_ref[:, idx], 0.5))


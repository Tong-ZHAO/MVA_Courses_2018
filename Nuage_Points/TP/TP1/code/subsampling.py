#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
from sklearn.preprocessing import label_binarize

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

from collections import Counter


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    keep_idx = np.arange(0, len(points), factor)
    decimated_points = points[keep_idx]
    decimated_colors = colors[keep_idx]
    decimated_labels = labels[keep_idx]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(voxel_size, points, colors = None, labels = None):

    min_coords, max_coords = points.min(0), points.max(0)
    pts_indices = np.floor((points - min_coords) / voxel_size)

    _, pts_grids = np.unique(pts_indices, return_inverse = True, axis = 0)
    num_grids = pts_grids.max() + 1 

    subsampled_points = []

    if colors is not None:
        subsampled_colors = []
    if labels is not None:
        subsampled_labels = []

    for i in range(num_grids):
        mask = (pts_grids == i)
        subsampled_points.append(points[mask].mean(0))
        if colors is not None:
            subsampled_colors.append(colors[mask].astype(float).mean(0).astype(np.uint8))
        if labels is not None:
            mode = Counter(labels[mask]).most_common(1)
            subsampled_labels.append(mode[0][0])

    if colors is None and labels is None:
        return np.array(subsampled_points)
    if colors is None:
        return np.array(subsampled_points), np.array(subsampled_labels) 
    if labels is None:
        return np.array(subsampled_points), np.array(subsampled_colors)  
    return np.array(subsampled_points), np.array(subsampled_colors), np.array(subsampled_labels) 


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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    
    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    
    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points = grid_subsampling(voxel_size, points)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_subsampled.ply', [subsampled_points], ['x', 'y', 'z'])
    
    print('Done')
    

    # Subsample the point cloud on a grid with color and label
    # It takes ~ 2 mins
    # ***********************************
    #
     
    if False:
        # Define the size of the grid
        voxel_size = 0.2

        # Subsample
        t0 = time.time()
        subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling(voxel_size, points, colors, labels)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

        # Save
        write_ply('../grid_subsampled_labels.ply', 
                [subsampled_points, subsampled_colors, subsampled_labels], 
                ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        
        print('Done')

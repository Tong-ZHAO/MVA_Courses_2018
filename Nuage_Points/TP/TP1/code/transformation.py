#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

def centralize_pts(pts):

    centroid = pts.mean(0).reshape((1, -1))
    new_pts = pts - centroid

    return new_pts, centroid


def scale_pts(pts, scale_factor = 2):

    return pts / scale_factor


def translate_pts(pts, offset):

    new_pts = pts + np.array(offset).reshape((1, 3))

    return new_pts


def rotate_pts(pts, axis = 2, angle = -90):
    # https://en.wikipedia.org/wiki/Transformation_matrix#Rotation_2

    unit_vec = np.zeros((3, 1))
    unit_vec[axis] = 1.

    cos_angle, sin_angle = np.cos(angle / 180 * np.pi), np.sin(angle / 180 * np.pi)

    trans_mat_1 = (1 - cos_angle) * unit_vec.dot(unit_vec.T)
    trans_mat_2 = sin_angle * np.array(([[0, -unit_vec[2, 0], unit_vec[1, 0]],
                                        [unit_vec[2, 0], 0, -unit_vec[0, 0]],
                                        [-unit_vec[1, 0], unit_vec[0, 0], 0]]))
    trans_mat_3 = np.eye(3) * cos_angle

    trans_mat = trans_mat_1 + trans_mat_2 + trans_mat_3

    return pts.dot(trans_mat.T)



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

    # Path of the file
    file_path = '../data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T

    # Get the scalar field which represent density as a vector
    density = data['scalar_density']

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #

    # Replace this line by your code
    transformed_points, centroid = centralize_pts(points)
    transformed_points = scale_pts(transformed_points, 2)
    transformed_points = rotate_pts(transformed_points, 2, -90)
    transformed_points = translate_pts(transformed_points, centroid)
    transformed_points = translate_pts(transformed_points, [0, -0.1, 0])

    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('../little_bunny.ply', [transformed_points, colors, density], ['x', 'y', 'z', 'red', 'green', 'blue', 'density'])

    print('Done')

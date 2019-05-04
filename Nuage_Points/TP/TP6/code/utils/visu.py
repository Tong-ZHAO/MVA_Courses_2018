#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Utility function given for visualization
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
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
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#


def show_ICP(data, ref, R_list, T_list, neighbors_list):
    '''
    Show a succession of transformation obtained by ICP.
    Inputs :
                  data = (d x N_data) matrix where "N_data" is the number of point and "d" the dimension
                   ref = (d x N_ref) matrix where "N_ref" is the number of point and "d" the dimension
                R_list = list of the (d x d) rotation matrices found at each iteration
                T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = list of the neighbors of data in ref at each iteration

    This function works if R_i and T_i represent the tranformation of the original cloud at iteration i, such
    that data_(i) = R_i * data + T_i.
    If you saved incremental transformations such that data_(i) = R_i * data_(i-1) + T_i, you will need to
    modify your R_list and T_list in your ICP implementation.
    '''

    # Get the number of iteration
    max_iter = len(R_list)

    # Get data dimension
    dim = data.shape[0]

    # Insert identity as first transformation
    R_list.insert(0, np.eye(dim))
    T_list.insert(0, np.zeros((dim, 1)))

    # Create global variable for the graph plot
    global iteration, show_neighbors
    iteration = 0
    show_neighbors = 0

    # Define the function drawing the points
    def draw_event():
        data_aligned = R_list[iteration].dot(data) + T_list[iteration]
        plt.cla()
        if dim == 2:
            ax.plot(ref[0], ref[1], '.')
            ax.plot(data_aligned[0], data_aligned[1], '.')
            if show_neighbors and iteration < max_iter:
                lines = [[data_aligned[:, ind1], ref[:, ind2]] for ind1, ind2 in enumerate(neighbors_list[iteration])]
                lc = mc.LineCollection(lines, colors=[0, 1, 0, 0.5], linewidths=1)
                ax.add_collection(lc)
            plt.axis('equal')
        if dim == 3:
            ax.plot(ref[0], ref[1], ref[2], '.')
            ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.')
            plt.axis('equal')
        if show_neighbors and iteration < max_iter:
            ax.set_title('Iteration {:d} ===> press right / left to change\nNeighbors ON ===> Press n to change (only in 2D)'.format(iteration))
        else:
            ax.set_title('Iteration {:d} ===> press right / left to change\nNeighbors OFF ===> Press n to change (only in 2D)'.format(iteration))

        plt.draw()

    # Define the function getting keyborad inputs
    def press(event):
        global iteration, show_neighbors
        if event.key == 'right':
            if iteration < max_iter:
                iteration += 1
        if event.key == 'left':
            if iteration > 0:
                iteration -= 1
        if event.key == 'n':
            if dim < 3:
                show_neighbors = 1 - show_neighbors
        draw_event()

    # Create figure
    fig = plt.figure()

    # Intitiate graph for 3D data
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        print('wrong data dimension')

    # Connect keyboard function to the figure
    fig.canvas.mpl_connect('key_press_event', press)

    # Plot first iteration
    draw_event()

    # Start figure
    plt.show()



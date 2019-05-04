import numpy as np
import matplotlib.pyplot as plt

def read_file(file_path):
    """Read data file from user defined path
    
    Params:
        file_path (str): the absolute path of the file to be read
        
    Returns:
        data (np.array): of size nxd, where n is the number of observations 
                         and d is the dimension of each observation
    """
    
    with open(file_path, "r") as fp:
        fp_line = fp.read().split("\n")[:-1]

    data = np.array([line.split() for line in fp_line]).astype(float)
    
    return data


def gaussian(X, mu, sigma):
    """Calculate the probabilistic function of a multivariate normal distribution given its parameters
    
    Params:
        X     (np.array): the observations, a matrix of size nxd
        mu    (np.array): the mean vector of size d
        sigma (np.array): the covariance matrix of size dxd
        
    Returns:
        X_proba (np.array): the probability of X_i belonging to the given normal distribution
    """
    
    k = X.shape[1]
    X_mean = X - mu
    
    denominator = np.sqrt((2 * np.pi) ** k * np.linalg.det(sigma))
    numerator   = -0.5 * np.sum(np.dot(X_mean, np.linalg.inv(sigma)) * X_mean, axis = 1)
    
    return np.squeeze(np.exp(numerator) / denominator)


def plot_cluster_EM_gaussian(data, y, mu, sigma, k = 4):
    """Function to draw the Gaussian contour at 90%.
   
    Params:
        data  (np.array): data set 
        y     (np.array): labels of the data
        mu    (np.array): list of mean vectors
        sigma (np.array): list of covariance matrices
        k     (int)     : number of clusters
    """
    
    plt.figure()
    plt.scatter(data[:,0], data[:,1], c = y)
    s = 4.6057
    
    for j in np.arange(0, k):   
        [V, D] = np.linalg.eig(sigma[j] * s)
        t = np.linspace(0,2 * np.pi)
        vec = np.array([np.cos(t), np.sin(t)])
        VD = D @ np.sqrt(np.diag(V))
        z = VD @ vec + mu[j].reshape(-1, 1)
        plt.plot(z[0, :], z[1, :])
        
    plt.grid()
    plt.show()


def plot_hmm_loss(loss_history):

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("t")
    plt.ylabel("MLE")
    plt.title("The MLE loss for HMM")
    plt.grid()
    plt.show()
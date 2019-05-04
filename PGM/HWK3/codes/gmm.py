import numpy as np
from utils import gaussian


def k_means_plus_plus(data, k, estim_time):
    """Implement the k-mean++ algorithm
    
    Params:
        data  (np.array): of size nxd.
        k          (int): number of clusters
        estim_time (int): number of simulations
        
    Returns:
        dict_results: the center, the labels and the loss history
    """
    
    # initialize results
    prev_center, prev_label, prev_loss = None, None, 1e10
    
    for i in range(estim_time):
        
        # random choose the first center
        k_init = data[np.random.randint(data.shape[0])].reshape((1, -1))
        
        # choose other centers one by one
        for i in range(1, k):
            dist = np.vstack([np.linalg.norm(data - k_center, axis = 1) for k_center in k_init])
            dist = dist[np.argmin(dist, axis = 0), np.arange(data.shape[0])]
            dist_sum = dist.sum() * np.random.random()
            
            new_idx = np.argmax(np.cumsum(dist) - dist_sum > 0)
            k_init = np.vstack((k_init, data[new_idx].reshape((1, -1))))
            
        # optimize center
        center, label, loss = k_means_oracle(data, k_init)
        
        if loss < prev_loss:
            prev_center, prev_label, prev_loss = center, label, loss
            
    return {"center": prev_center, "label": prev_label, "loss": prev_loss}


def k_means_oracle(data, center):
    """ Update the k-means model for 1 time
    
    Params:
        data   (np.array): of size nxd
        center (np.array): of size kxd
    
    Returns:
        center (np.array): new centers
        label  (np.array): new labels
        loss   (np.array): new l2 loss
    """
    n, k = data.shape[0], center.shape[0]
    prev_loss = 1e10
    
    while True:
        dist = np.vstack([np.linalg.norm(data - k_center, axis = 1) for k_center in center])
        label = np.argmin(dist, axis = 0)
        loss = np.sum(dist[label, np.arange(n)])
        
        if loss < prev_loss:
            prev_loss = loss
            center = np.vstack(np.mean(data[label == i], axis = 0) for i in range(k))
        else:
            break
            
    return center, label, prev_loss


def EM_gaussian_E_step(data, alpha, mu, sigma):
    """ Function that computes the E-step of the EM algorithm
    
    Params:
        data: data set 
        alpha= proportion of each Gaussian.
        mu= list with mean values
        sigma= list with covariate matrices
    
    Returns:
        Tau: matrix with the probabilities P(z=j|x,tetha)
    """
    tau = np.zeros((data.shape[0],len(alpha)))
    for j in np.arange(0, len(alpha)):
        tau[:,j] = alpha[j] * gaussian(data, mu[j], sigma[j])
    denominator = tau.sum(1)
    for j in np.arange(0, len(alpha)):
        tau[:, j] = tau[:, j] / denominator
    return tau
 
    
def EM_gaussian_M_step(data, tau):
    """Funtion that computes the M-step value of the EM algorithm
    
    Params:
        data: data set
        tau: matrix with the probabilities P(z=j|x,tetha)
    
    Returns:
        alpha= proportion of each Gaussian.
        mu= list with mean values
        sigma= list with covariate matrices
    """
    sigma = []
    mu = []
    alpha = np.zeros(tau.shape[1])
    for j in np.arange(0, tau.shape[1]):
        alpha[j] = np.mean(tau[:, j])
        mu.append(np.apply_along_axis(lambda x: np.average(x,weights = tau[:,j]), 0, data))
        
        W = np.identity(data.shape[0])
        np.fill_diagonal(W, tau[:, j])
        data_centered = data - mu[j]
        sigma.append((np.transpose(data_centered).dot(W).dot(data_centered)) / np.sum(tau[:,j]))
    return dict(alpha = alpha, mu = mu, sigma = sigma)


def EM_gaussian(data,max_iter=100,k=2):
    """Function that performs the EM algorithm
    
    Params:
        data: data set 
        max_iter: maximum number of iteractions.
        k: number of clusters to find
    
    Returns:
        alpha: proportion of each Gaussian.
        mu: list with mean values
        sigma: list with covariate matrices
        tau: matrix with the probabilities P(z=j|x,tetha)
        loglike: sequence of loglikehood.
    """
    
    # K means initialization
    kmeans = k_means_plus_plus(data, k, 20)
    labels = kmeans["label"]
    
    loglike_seq = np.zeros(max_iter + 1)
    mu = []
    sigma = []
    alpha = np.zeros(k)
    for i in np.arange(0, k):
        index = np.where(labels == i)[0]
        mu.append(np.apply_along_axis(np.mean, 0, data[index,:]))
        sigma.append(np.cov(data[index, 0], data[index, 1]))
        alpha[i] = len(index) / data.shape[0]
    n_iter = 0
    loglike_seq[n_iter] = MixtureGaus_loglike(data,alpha,mu,sigma)
    
    while (n_iter < max_iter):
        # E-step
        tau = EM_gaussian_E_step(data, alpha, mu, sigma)
        # M-step
        results = EM_gaussian_M_step(data, tau)
        mu = results["mu"]
        alpha = results["alpha"]
        sigma = results["sigma"]
        n_iter = n_iter + 1
        loglike_seq[n_iter] = MixtureGaus_loglike(data, alpha, mu, sigma)
        
    return dict(alpha = alpha,mu = mu,sigma = sigma,tau = tau,loglike = loglike_seq)


def MixtureGaus_loglike(data, alpha, mu, sigma):
    """
    Params:
        data: data set
        alpha: proportion of each Gaussian.
        mu: list with mean values
        sigma: list with covariate matrices
    
    Returns:
        loglike: loglikelihood of the observations
    """
    
    aux_loglike = np.zeros((data.shape[0], len(alpha)))
    
    for j in np.arange(0,len(alpha)):
        aux_loglike[:,j] = alpha[j] * gaussian(data, mu[j], sigma[j])
        
    aux_loglike = np.apply_along_axis(lambda x: np.log(np.sum(x)), 1, aux_loglike)
    loglike = np.sum(aux_loglike) / float(len(data))
    
    return loglike
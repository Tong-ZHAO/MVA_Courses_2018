import numpy as np

from copy import deepcopy
from itertools import product
from scipy.special import logsumexp
from utils import gaussian


class HMM_LOG:
    
    def __init__(self, k, obs, init_mus, init_sigmas, init_z0 = None, init_A = None):
        """ This class implements a hidden markov model. The EM algorithm is calculated on log scale.
        
        Params:
            k   (int)     : number of clusters
            obs (np.array): a matrix of size n x m, where: n is the number of 
                            observations and m is the dim of the observation

            init_mus    ([np.array]): initial centers of each cluster
            init_sigmas ([np.array]): initial variance matrix of each cluster
            init_z0     (np.array)  : initial distribution of z0
            init_A      (np.array)  : initial transition matrix
            
        Attris:
            k (int): number of clusters
            n (int): number of observations
            d (int): dim of observations
            
            obs (np.array): observations
            
            mu    (np.array): the centers of each cluster
            sigma (np.array): the variance matrix of each cluster
            l     (float)   : current estimated log-likelihood
            
            log_z0 (np.array): the initial distribution of z0 on log scale
            log_A  (np.array): the transition matrix on log scale
            log_F  (np.array): the emission proba on log scale
            
            log_alpha (np.array): forward proba on log scale
            log_beta  (np.array): backward proba on log scale
            
            preds (np.array): the sequence of more likely states   
        """
        
        self.k = k
        self.n = obs.shape[0]
        self.d = obs.shape[1]

        self.obs = obs
        
        self.mu = deepcopy(init_mus)
        self.sigma = deepcopy(init_sigmas)

        self.init_params()
        if type(init_z0) != type(None):
            self.log_z0 = np.log(init_z0)
        if type(init_A) != type(None):
            self.log_A = np.log(init_A)
        
        self.update_emission()
        self.compute_log_likelihood(first_time = True)
        
        
    def init_params(self):
        """
        Initialize z0 to be [1/k, ..., 1/k] and A to be 1/k * ones((k, k))
        
        Only the log scale is saved.
        """
        
        # z0
        z0 = np.ones((self.k)) / float(self.k)
        self.log_z0 = np.log(z0)
        
        # transition matrix
        A = np.ones((self.k, self.k)) / self.k
        self.log_A = np.log(A)
        
        
    def update_emission(self):
        """
        Calculate the emission matrix given the estimated mus and sigmas
        
        Only the log scale is saved.
        """
        
        F = np.zeros((self.n, self.k))

        for i in range(self.k):
            F[:, i] = gaussian(self.obs, self.mu[i], self.sigma[i])
            
        self.log_F = np.log(F)
        
        
    def log_alpha_recursion(self, t):
        """
        Calculate the forward alpha message by a recursion formula.
        
        Log scale is used to prevent vanishing message. 
        """
        
        if t == 0:
            return (self.log_z0 + self.log_F[0]).reshape((1, -1))
        
        histo_alpha = self.log_alpha_recursion(t - 1)
        prev_alpha = histo_alpha[-1].reshape((1, -1))
        proba = prev_alpha + self.log_A
        
        curr_alpha = self.log_F[t] + proba.max(1) + logsumexp(proba - proba.max(1, keepdims = True), axis = 1)
        
        return np.vstack((histo_alpha, curr_alpha.reshape((1, -1))))
    
    
    def log_beta_recursion(self, t):
        """
        Calculate the backward beta message by a recursion formula.
        
        Log scale is used to prevent vanishing message. 
        """
        
        if t == self.n - 1:
            return np.zeros((1, self.k))
        
        histo_beta = self.log_beta_recursion(t + 1)
        prev_beta = histo_beta[0]
        proba = (prev_beta + self.log_F[t + 1]).reshape((-1, 1)) + self.log_A
        
        curr_beta = proba.max(0) + logsumexp(proba - proba.max(0, keepdims = True), axis = 0)
        
        return np.vstack((curr_beta.reshape((1, -1)), histo_beta))
    
    
    def smoothing(self):
        """
        Calculate the probability P(z_t|y_0, ..., y_T)
        """
        
        gamma = self.log_alpha + self.log_beta
        gamma_max = gamma.max(1)
        gamma_sum = logsumexp(gamma - gamma_max[:, np.newaxis], axis = 1) + gamma_max
        gamma = gamma - gamma_sum.reshape((-1, 1))
            
        return np.exp(gamma), gamma_sum
    
    
    def smoothing_pair(self, gamma_sum):
        """
        Calculate the probability P(z_t+1, z_t|y_0, ..., y_T)
        """

        epsilon = np.zeros((self.n - 1, self.k ,self.k))
        
        for (i, j) in product(range(self.k), range(self.k)):
            epsilon[:, j, i] = self.log_alpha[:-1, i] + self.log_beta[1:, j] + self.log_A[j, i] + self.log_F[1:, j]

        epsilon = np.exp(epsilon - gamma_sum[:-1, np.newaxis, np.newaxis])

        return epsilon
    
    
    def update_E_step(self):
        
        self.log_alpha = self.log_alpha_recursion(self.n - 1)
        self.log_beta = self.log_beta_recursion(0)
        
        gamma, gamma_sum = self.smoothing()
        epsilon = self.smoothing_pair(gamma_sum)
        
        
        
        return gamma, gamma_sum ,epsilon 
    
    
    def update_M_step(self,gamma,gamma_sum,epsilon):
        """
        M step given observations
        """
        
        #self.log_alpha = self.log_alpha_recursion(self.n - 1)
        #self.log_beta = self.log_beta_recursion(0)
        
        gamma, gamma_sum = self.smoothing()
        epsilon = self.smoothing_pair(gamma_sum)
        
        z0 = gamma[0]
        A = epsilon.sum(0) / np.sum(epsilon, axis = (0, 1))
    
        # update gaussian params
        for i in range(self.k):
            gamma_sum = gamma[:, i].sum()
            self.mu[i] = (gamma[:, i].reshape((-1, 1)) * self.obs).sum(0) / gamma_sum
            obs_mean = self.obs - self.mu[i]
            self.sigma[i] = np.dot(np.dot(obs_mean.T, np.diag(gamma[:, i])), obs_mean) / gamma_sum
            
        # update emission distribution
        self.log_z0 = np.log(z0)
        self.log_A = np.log(A)
        self.update_emission()
        
        
    def inference(self):
        """
        Estimate the sequence of more likely states given the estimated parameters
        """
        
        path = np.zeros((self.n, self.k))
        prev_proba = self.log_z0 + self.log_F[0]
        
        for i in range(1, self.n):
            trasp_proba = prev_proba.reshape((-1, 1)) + self.log_A
            prev_proba = self.log_F[i] + np.max(trasp_proba, axis = 0)
            path[i - 1] = np.argmax(trasp_proba, axis = 0)

        states = [np.argmax(prev_proba)]

        for i in range(self.n - 2, -1, -1):        
            states.append(path[i, int(states[-1])])
            
        return np.array(states[::-1]).astype(int)
    
    
    def train(self, max_iter = 20, tol = 1e-6):
        """
        EM algorithm for HMM
        """
        
        self.loss = [-np.inf]
        
        for i in range(max_iter):
            gamma, gamma_sum ,epsilon= self.update_E_step()
            self.update_M_step(gamma, gamma_sum ,epsilon)
            self.preds = self.inference()
            self.compute_log_likelihood()
            self.loss.append(self.l)
            if self.loss[-1] < self.loss[-2] + tol:
                break
    
        
    def compute_log_likelihood(self,first_time=False):
        """
        Calculate the likelihood function 
        """
        if first_time:
            self.log_alpha = self.log_alpha_recursion(self.n - 1)
            self.log_beta = self.log_beta_recursion(0)

        log_l = self.log_alpha[self.n - 1] + self.log_beta[self.n - 1]
        lmax = log_l.max(0)

        self.l = lmax + logsumexp(log_l - lmax)
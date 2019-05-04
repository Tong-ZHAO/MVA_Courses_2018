import numpy as np
from tqdm import tqdm_notebook as tqdm

def estim_theta(inv_term, multi_term):

    return np.dot(np.linalg.inv(inv_term), multi_term).reshape((-1, 1))

def upper_bound(inv_term, features):

    inv_term = np.linalg.inv(inv_term) 
    sqrt_term = np.dot(np.dot(features, inv_term), features.T)

    return np.diagonal(np.sqrt(sqrt_term)).reshape((-1, 1))


def optim_oracle(model, N, T, method = "LinUCB", lambd = 1, alpha = 1, epsilon = .2):

    num_features, num_actions = model.n_features, model.n_actions
    error_thetas, accum_regrets = np.zeros((N, T)), np.zeros((N, T))
    max_rew = model.best_arm_reward()

    for k in tqdm(range(N), desc="Simulating {}".format(method)):

        if method == "LinUCB":
            rew, thetas = LinUCB(model, T, alpha, lambd)
        elif method == "Random":
            rew, thetas = RandomPolicy(model, T, lambd)
        elif method == "Greedy":
            rew, thetas = EpsilonPolicy(model, T, epsilon, lambd)
        else:
            raise KeyError("The method %s have not been implemented!" % method)

        error_thetas[k] = np.linalg.norm(thetas - model.real_theta, axis = 1)
        accum_regrets[k] = np.arange(1, T + 1) * max_rew - np.cumsum(rew)

    return error_thetas.mean(0), accum_regrets.mean(0)
        
            


def LinUCB(model, T, alpha, lambd):

    num_features, num_actions = model.n_features, model.n_actions
    rew, thetas = np.zeros((T, 1)), np.zeros((T, num_features))

    inv_term, multi_term = lambd * np.eye(num_features), np.zeros(num_features)

    for t in range(T):

        theta_hat = estim_theta(inv_term, multi_term)
        a_t = np.argmax(np.dot(model.features, theta_hat) + alpha * upper_bound(inv_term, model.features))
        r_t = model.reward(a_t) # get the reward

        # update algorithm
        rew[t] = r_t
        thetas[t] = np.squeeze(theta_hat)

        # update params
        feature = model.features[a_t].reshape((-1, 1))
        inv_term += np.dot(feature, feature.T)
        multi_term += r_t * np.squeeze(feature)

    return rew, thetas


def RandomPolicy(model, T, lambd):

    num_features, num_actions = model.n_features, model.n_actions
    rew, thetas = np.zeros((T, 1)), np.zeros((T, num_features))

    inv_term, multi_term = lambd * np.eye(num_features), np.zeros(num_features)

    for t in range(T):

        theta_hat = estim_theta(inv_term, multi_term)
        a_t= np.random.randint(num_actions)
        r_t = model.reward(a_t) # get the reward

        # update algorithm
        rew[t] = r_t
        thetas[t] = np.squeeze(theta_hat)

        # update params
        feature = model.features[a_t].reshape((-1, 1))
        inv_term += np.dot(feature, feature.T)
        multi_term += r_t * np.squeeze(feature)

    return rew, thetas


def EpsilonPolicy(model, T, epsilon, lambd):

    num_features, num_actions = model.n_features, model.n_actions

    rew, thetas = np.zeros((T, 1)), np.zeros((T, num_features))
    inv_term, multi_term = lambd * np.eye(num_features), np.zeros(num_features)
    
    max_rate_index = np.argmax(model.features.mean(0))

    for t in range(T):

        theta_hat = estim_theta(inv_term, multi_term)
        a_t = np.random.randint(num_actions) if np.random.rand() < epsilon else max_rate_index
        r_t = model.reward(a_t) # get the reward

        # update algorithm
        rew[t] = r_t
        thetas[t] = np.squeeze(theta_hat)

        # update params
        feature = model.features[a_t].reshape((-1, 1))
        inv_term += np.dot(feature, feature.T)
        multi_term += r_t * np.squeeze(feature)

    return rew, thetas




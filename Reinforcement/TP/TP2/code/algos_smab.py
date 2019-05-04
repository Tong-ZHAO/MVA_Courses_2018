import numpy as np


def NAIVE(T, MAB):
    """

    Param:
        T   (int)         : The time of simulation times
        MAB (list of arms): The list of arms objects    
    """

    K = len(MAB)
    assert(T > K), "The simulation times should be larger than the number of arms!"

    N, S = np.zeros(K), np.zeros(K)
    rew, draws = [], []

    # Initialization phase
    for t in range(K):
        S[t] = MAB[t].sample()
        N[t] += 1
        rew.append(S[t])
        draws.append(t)

    for t in range(K, T):
        index = np.argmax(S / N)
        reward = MAB[index].sample()
        S[index] += reward
        N[index] += 1
        rew.append(reward)
        draws.append(index)

    return np.array(rew), np.array(draws)




def UCB1(T, MAB, rho = 1):
    """Implement the UCB1 algorithm to simulate a bandit game

    Param:
        T   (int)         : The time of simulation times
        MAB (list of arms): The list of arms objects
    """

    K = len(MAB)
    assert(T > K), "The simulation times should be larger than the number of arms!"

    N, S = np.zeros(K), np.zeros(K)
    rew, draws = [], []

    # Initialization phase
    for t in range(K):
        S[t] = MAB[t].sample()
        N[t] += 1
        rew.append(S[t])
        draws.append(t)

    for t in range(K, T):
        scores = S / N + rho * np.sqrt(np.log(t) / (2 * N))
        index = np.argmax(scores)
        reward = MAB[index].sample()
        S[index] += reward
        N[index] += 1
        rew.append(reward)
        draws.append(index)

    return np.array(rew), np.array(draws)


def TS(T, MAB, multi = True):

    K = len(MAB)

    S, N = np.zeros(K), np.zeros(K)
    rew, draws = [], []

    for t in range(T):
        scores = np.random.beta(S + 1, N - S + 1)
        index = np.argmax(scores)
        # deal with multi-class cases
        reward = np.random.binomial(1, MAB[index].sample()) if multi else  MAB[index].sample()
        S[index] += reward
        N[index] += 1
        rew.append(reward)
        draws.append(index)

    return np.array(rew), np.array(draws)


def expected_regret(N, T, MAB, method, multi = False):

    max_rew = max([arm.mean for arm in MAB])
    sim_regrets = np.zeros((N, T))

    for i in range(N):

        if method == "ucb1":
            rew, _ = UCB1(T, MAB)
        elif method == "thompson":
            rew, _ = TS(T, MAB, multi)
        elif method == "naive":
            rew, _ = NAIVE(T, MAB)
        else:
            raise KeyError("The method %s have not been implemented!" % method)

        regret = np.arange(1, T + 1) * max_rew - np.cumsum(rew)
        sim_regrets[i] = regret

    return sim_regrets.mean(axis = 0)


def kl_divergence(X, Y):

    return X * np.log(X / Y) + (1 - X) * np.log((1 - X) / (1 - Y))


def complexity(MAB):

    p_a = [arm.mean for arm in MAB]
    p_star = max(p_a)
    p_a = np.array([p for p in p_a if p < p_star])

    compx = np.sum((p_star - p_a) / kl_divergence(p_a, p_star))

    return compx




    





    


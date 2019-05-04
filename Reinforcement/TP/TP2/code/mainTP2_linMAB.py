import numpy as np
from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
import matplotlib.pyplot as plt
from tqdm import tqdm

random_state = np.random.randint(0, 24532523)
model = ToyLinearModel(
    n_features=8,
    n_actions=20,
    random_state=random_state,
    noise=0.1)

# model = ColdStartMovieLensModel(
#     random_state=random_state,
#     noise=0.1
# )

n_a = model.n_actions
d = model.n_features

T = 6000


nb_simu = 50 # you may want to change this!

##################################################################
# define the algorithms
# - Random
# - Linear UCB
# - Eps Greedy
# and test it!
##################################################################

regret = np.zeros((nb_simu, T))
norm_dist = np.zeros((nb_simu, T))

for k in tqdm(range(nb_simu), desc="Simulating {}".format(alg_name)):

    for t in range(T):
        a_t = ...  # algorithm picks the action
        r_t = model.reward(a_t) # get the reward

        # do something (update algorithm)

        # store regret
        regret[k, t] = model.best_arm_reward() - r_t
        norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

# compute average (over sim) of the algorithm performance and plot it
mean_norms = ...
mean_regret = ...

plt.figure(1)
plt.subplot(121)
plt.plot(mean_norms, label=alg_name)
plt.ylabel('d(theta, theta_hat)')
plt.xlabel('Rounds')
plt.legend()

plt.subplot(122)
plt.plot(mean_regret.cumsum(), label=alg_name)
plt.ylabel('Cumulative Regret')
plt.xlabel('Rounds')
plt.legend()
plt.show()

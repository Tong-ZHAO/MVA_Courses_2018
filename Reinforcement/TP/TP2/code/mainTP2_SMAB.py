import numpy as np
import arms
import matplotlib.pyplot as plt
from ucb import UCB1, TS

# Build your own bandit problem

# this is an example, please change the parameters or arms!
arm1 = arms.ArmBernoulli(0.30, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBernoulli(0.25, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmBernoulli(0.20, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmBernoulli(0.10, random_state=np.random.randint(1, 312414))

MAB = [arm1, arm2, arm3, arm4]

# bandit : set of arms

nb_arms = len(MAB)
means = [el.mean for el in MAB]

# Display the means of your bandit (to find the best)
print('means: {}'.format(means))
mu_max = np.max(means)

# Comparison of the regret on one run of the bandit algorithm
# try to run this multiple times, you should observe different results

T = 5000  # horizon

rew1, draws1 = UCB1(T, MAB)
reg1 = mu_max * np.arange(1, T + 1) - np.cumsum(rew1)
rew2, draws2 = TS(T, MAB)
reg2 = mu_max * np.arange(1, T + 1) - np.cumsum(rew2)

# reg3 = naive strategy

# add oracle t -> C(p)log(t)

plt.figure(1)
x = np.arange(1, T+1)
plt.plot(x, reg1, label='UCB')
plt.plot(x, reg2, label='Thompson')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.legend()

plt.show()

# (Expected) regret curve for UCB and Thompson Sampling

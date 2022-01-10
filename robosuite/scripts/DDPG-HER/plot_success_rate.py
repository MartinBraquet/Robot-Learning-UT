import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

env_name = 'Fetch Pick and Place'

success_rate = genfromtxt('pick-place-success-rate.csv', delimiter=',')
success_rate_mean = np.zeros(50)
for i in range(50):
    success_rate_mean[i] = np.mean(success_rate[10*i:10*(i+1)])

success_rate_random = np.zeros(len(success_rate_mean))

plt.plot(success_rate_mean, label='DDPG-HER')
plt.plot(success_rate_random, label='Random')
plt.xlabel('# episodes')
plt.ylabel('Success Rate')
plt.suptitle(env_name, fontsize=14)
plt.ylim(0, 1)
plt.legend()
plt.savefig(env_name + '.png')
plt.show()
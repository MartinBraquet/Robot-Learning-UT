import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

env_name = 'Fetch Reach'

success_rate = np.load('tot_reward_list.npy')
success_rate_mean = []
n = 10
i = 0
j = i + n
while j < len(success_rate):
    success_rate_mean.append(np.mean(success_rate[i:j]))
    i = i + n
    j = j + n
success_rate_mean = np.array(success_rate_mean)

plt.plot(n*np.arange(len(success_rate_mean)), success_rate_mean, label='DDPG-HER')
plt.xlabel('# episodes')
plt.ylabel('Reward')
plt.suptitle(env_name, fontsize=14)
plt.legend()
plt.savefig(env_name + '.png')
plt.show()

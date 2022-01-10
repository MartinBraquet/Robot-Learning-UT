from __future__ import division
import gym
import numpy as np
import torch
from torch._C import dtype
from torch.autograd import Variable
import os
import psutil
import gc

import buffer
import train

MAX_EPISODES = 5
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
EPS_PER_SAVE = 50
ENV_STRING = 'FetchReach-v1'

print('Start TEST')

if ENV_STRING in ['FetchPickAndPlace-v1', 'FetchReach-v1']:
	env = gym.make(ENV_STRING,reward_type='dense')
	S_DIM = env.observation_space['observation'].shape[0]+env.observation_space['desired_goal'].shape[0]
else:
	env = gym.make(ENV_STRING)
	S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)
print(' Observation Space :- ', env.observation_space)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
offset = trainer.load_models(-1,env_string=ENV_STRING)
for _ep in range(MAX_EPISODES):
	observation = env.reset()
	print('EPISODE :- ', _ep)
	tot_reward = 0.0
	for r in range(MAX_STEPS):
		env.render()
		if ENV_STRING in ['FetchPickAndPlace-v1', 'FetchReach-v1']:
			#state=np.float32(observation['observation'])
			state = np.concatenate((observation['observation'], observation['desired_goal']),dtype=np.float32)
		else:
			state = np.float32(observation)

		action = trainer.get_exploration_action(state)

		observation, reward, done, info = env.step(action)

		# print(observation)

		tot_reward = tot_reward + reward
		if done:
			break
	print('Total Reward: ', tot_reward)

	# check memory consumption and clear memory
	gc.collect()

print('Completed test')

from __future__ import division
import gym
import numpy as np
import torch
from torch._C import dtype
from torch.autograd import Variable
import os
import psutil
import gc

import train
import buffer

#env = gym.make('BipedalWalker-v2') 'Pendulum-v1'

MAX_EPISODES = 3000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
EPS_PER_SAVE = 1000
ENV_STRING = 'FetchReach-v1'
RENDER_ENV = False

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

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
offset = trainer.load_models(-1,env_string=ENV_STRING)

tot_reward_list = []
for _ep in range(MAX_EPISODES):
	observation = env.reset()
	print('EPISODE :- ', _ep)
	tot_reward = 0.0
	for r in range(MAX_STEPS):
		if RENDER_ENV:
			env.render()

		if ENV_STRING in ['FetchPickAndPlace-v1', 'FetchReach-v1']:
			#state=np.float32(observation['observation'])
			state = np.concatenate((observation['observation'], observation['desired_goal']),dtype=np.float32)
		else:
			state = np.float32(observation)

		action = trainer.get_exploration_action(state)

		new_observation, reward, done, info = env.step(action)

		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue

		if done:
			new_state = None
		else:
			if ENV_STRING in ['FetchPickAndPlace-v1', 'FetchReach-v1']:
				#new_state = np.float32(new_observation['observation'])
				new_state = np.concatenate((new_observation['observation'], new_observation['desired_goal']),dtype=np.float32)
			else:
				new_state = np.float32(new_observation)
			# push this exp in ram
			ram.add(state, action, reward, new_state)

		observation = new_observation

		# perform optimization
		trainer.optimize()
		tot_reward = tot_reward + reward
		if done:
			break
	tot_reward_list.append(tot_reward)
	print('Total Reward: ', tot_reward)

	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	if _ep%EPS_PER_SAVE== 0:
		trainer.save_models(_ep+offset,ENV_STRING)

trainer.save_models(MAX_EPISODES+offset,ENV_STRING)
print('Completed episodes')
tot_reward_list = np.array(tot_reward_list)
np.save('tot_reward_list.npy', tot_reward_list)

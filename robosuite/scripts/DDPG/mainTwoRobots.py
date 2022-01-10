from __future__ import division
import gym
import numpy as np
from numpy.random import triangular
import torch
from torch._C import dtype
from torch.autograd import Variable
import os
import psutil
import gc

import trainTwoAgents # new trainer (TODO)
import train_v2
import buffer
import robosuite as suite
from robosuite.wrappers import GymWrapper

MAX_EPISODES = 1000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
EPS_PER_SAVE = 25
ENV_STRING = 'robosuite'
TRAINING = True
RENDER_ENV = False
RESET_AGENT = False

env = suite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    #controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=RENDER_ENV,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    reward_shaping=True,                    # use dense rewards
    use_object_obs=False,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
)
S_DIM = 3+7
A_DIM = 8
A_MAX = 1

print(env.reset())

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = trainTwoAgents.TrainerTwoAgents(S_DIM, A_DIM, A_MAX, ram)
#offset = trainer.load_models(1000,env_string=ENV_STRING)
offset = 0
for _ep in range(MAX_EPISODES):
	observation = env.reset()
	print('EPISODE :- ', _ep)
	tot_reward = 0.0
	for r in range(MAX_STEPS):
		if RENDER_ENV:
			env.render()
		
		joint_c = observation['robot0_joint_pos_cos']
		joint_s = observation['robot0_joint_pos_sin']
		#print('Joint Cosines, ', joint_c)
		#print('Joint Sines, ', joint_s)
		joint_angs = np.arctan2(joint_s,joint_c)
		state = np.concatenate((observation['cube_pos'], joint_angs),dtype=np.float32)

		if TRAINING == True:
			action = trainer.get_exploration_action(state)
		else:
			action = trainer.get_exploitation_action(state)


		new_observation, reward, done, info = env.step(action)

		if done:
			new_state = None
		else:
			joint_c = new_observation['robot0_joint_pos_cos']
			joint_s = new_observation['robot0_joint_pos_sin']
			joint0_angs = np.arctan2(joint_s,joint_c)
			joint_c = new_observation['robot1_joint_pos_cos']
			joint_s = new_observation['robot1_joint_pos_sin']
			joint1_angs = np.arctan2(joint_s,joint_c)
			#print('Joint Cosines, ', joint_c)
			#print('Joint Sines, ', joint_s)
			new_state = np.concatenate((new_observation['cube_pos'], joint0_angs, joint1_angs),dtype=np.float32)
			# push this exp in ram
			ram.add(state, action, reward, new_state)

		observation = new_observation

		# perform optimization
		if TRAINING == True:
			trainer.optimize()

		tot_reward = tot_reward + reward
		if done:
			break
	print('Total Reward: ', tot_reward)

	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	if (_ep%EPS_PER_SAVE== 0) and (TRAINING == True):
		trainer.save_models(_ep+offset,ENV_STRING)

trainer.save_models(MAX_EPISODES+offset,ENV_STRING)
print('Completed episodes')

from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import gc
import os
import argparse
import h5py

import train
import train_v2
import train_v3
import buffer
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config

def run_IMESB(args):
	MAX_BUFFER = 1000000
	ENV_STRING = 'robosuite-v3'

	S_DIM = 3+7
	A_DIM = 7
	A_MAX = 1

	config = load_controller_config(default_controller=args.CTRL_STRING)
	env =suite.make(
		"Lift",
		robots="Sawyer",                # use Sawyer robot
		use_camera_obs=False,           # do not use pixel observations
		has_offscreen_renderer=False,   # not needed since not using pixel obs
		has_renderer=args.RENDER_ENV,        # make sure we can render to the screen
		reward_shaping=True,            # use dense rewards
		control_freq=20,                # control should happen fast enough so that simulation looks smooth
		controller_configs = config,	# Controller config
	)
	ram = buffer.MemoryBuffer(MAX_BUFFER)
	if ENV_STRING == 'robosuite-v2':
		trainer = train_v2.Trainer_v2(S_DIM,1.0,A_DIM,A_MAX,ram,1)
	elif ENV_STRING == 'robosuite-v3':
		trainer = train_v2.Trainer_v2(S_DIM,1.0,A_DIM,A_MAX,ram,1)
	else:
		trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

	if args.RESET_AGENT:
		offset = 0
	else:
		offset = trainer.load_models(-1,env_string=ENV_STRING)

	out_dir = "/home/spatric5/repos/robosuite/robosuite/models/assets/demonstrations/1638308941_8885477"
	hdf5_path = os.path.join(out_dir, "demo.hdf5")
	f = h5py.File(hdf5_path, "r")
	for demo_num in f['data'].keys():
		print('Adding ', demo_num, ' to training dataset')
		states = f['data'][demo_num]['states']
		actions = f['data'][demo_num]['actions']
		#if(demo_num == 'demo_3') or (demo_num == 'demo_4'):
		#	print('Skipping')
		#	continue
		env.reset()
		for act, stat in zip(actions, states):
			env.sim.set_state_from_flattened(stat)
			env.sim.forward()
			observation = env._get_observations()
			joint_c = observation['robot0_joint_pos_cos']
			joint_s = observation['robot0_joint_pos_sin']
			joint_angs = np.arctan2(joint_s,joint_c)
			state = np.concatenate((observation['cube_pos'], joint_angs),dtype=np.float32)
			if args.RENDER_ENV:
				env.render()
			new_observation, reward, _, _ = env.step(act)
			joint_c = new_observation['robot0_joint_pos_cos']
			joint_s = new_observation['robot0_joint_pos_sin']
			joint_angs = np.arctan2(joint_s,joint_c)
			new_state = np.concatenate((new_observation['cube_pos'], joint_angs),dtype=np.float32)

			# push this exp in ram
			ram.add(state, act, reward, new_state)
		
			if args.TRAINING == True:
				trainer.optimize()
		gc.collect()
	

	f.close()

	for _ep in range(args.MAX_EPISODES):
		observation = env.reset()
		print('EPISODE :- ', _ep)
		tot_reward = 0.0
		for r in range(args.MAX_STEPS):
			if args.RENDER_ENV:
				env.render()
			
			joint_c = observation['robot0_joint_pos_cos']
			joint_s = observation['robot0_joint_pos_sin']
			joint_angs = np.arctan2(joint_s,joint_c)
			state = np.concatenate((observation['cube_pos'], joint_angs),dtype=np.float32)

			if (args.TRAINING == True) or (_ep % 10 != 0):
				action = trainer.get_exploration_action(state)
			else:
				action = trainer.get_exploitation_action(state)
			
			new_observation, reward, done, info = env.step(action)

			if done:
				new_state = None
			else:
				joint_c = new_observation['robot0_joint_pos_cos']
				joint_s = new_observation['robot0_joint_pos_sin']
				joint_angs = np.arctan2(joint_s,joint_c)
				new_state = np.concatenate((new_observation['cube_pos'], joint_angs),dtype=np.float32)

				# push this exp in ram
				ram.add(state, action, reward, new_state)

			observation = new_observation

			# perform optimization
			if args.TRAINING == True:
				trainer.optimize()

			tot_reward = tot_reward + reward
			if done:
				break
		print('Total Reward: ', tot_reward)

		# check memory consumption and clear memory
		gc.collect()

		if (_ep%args.EPS_PER_SAVE== 0) and (args.TRAINING == True):
			trainer.save_models(_ep+offset,ENV_STRING)

	trainer.save_models(args.MAX_EPISODES+offset,ENV_STRING)
	print('Completed episodes')

if __name__ == "__main__":
	# Parse Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--Max_Eps",dest='MAX_EPISODES',action='store',type=int,
						default=1200,help = "Maximum Number of Episodes")
	parser.add_argument("--Max_Steps",dest='MAX_STEPS',action='store',type=int,
						default=100,help = "Maximum Number of Time Steps per Episode")
	parser.add_argument("--controller",dest='CTRL_STRING',action='store',
						default='OSC_POSE',help = "Controller String: OSC_POSE, OSC_POSITION, JOINT_POSITION ...")
	parser.add_argument("--Render",dest='RENDER_ENV',action='store_true',
						default=False,help='Render robosuite environment')
	parser.add_argument("--Reset",dest='RESET_AGENT',action='store_true',
						default=False,help='Reset learned agent model')
	parser.add_argument("--Testing",dest='TRAINING',action='store_false',
						default=True,help='Run evaluation and no training')
	parser.add_argument("--Eps_BW_Save",dest='EPS_PER_SAVE',action='store',type=int,
						default=30,help='Number of episodes until saving model')
	parser.add_argument("--Expirment",dest='RUN_TYPE',action='store',type=int,choices=[1,2,3],
						default=3,help='What is being trained. 1 - Estimator for agent 1. 2 = Estimator for agent 2. 3 - Policy and Joint Estimator')
	parser.add_argument("--Use_Replays",dest='USE_REPLAYS',action='store_true',
						default=False,help='Use human demonstrations to initialize agent')
	args = parser.parse_args()

	run_IMESB(args)
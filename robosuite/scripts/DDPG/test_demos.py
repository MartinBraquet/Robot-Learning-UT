import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np
import json

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action

def gather_demonstrations_as_hdf5(env):
	"""
	Gathers the demonstrations saved in @directory into a
	single hdf5 file.

	The strucure of the hdf5 file is as follows.

	data (group)
		date (attribute) - date of collection
		time (attribute) - time of collection
		repository_version (attribute) - repository version used during collection
		env (attribute) - environment name on which demos were collected

		demo1 (group) - every demonstration has a group
			model_file (attribute) - model xml string for demonstration
			states (dataset) - flattened mujoco states
			actions (dataset) - actions applied during demonstration

		demo2 (group)
		...
	"""
	out_dir = "/home/spatric5/repos/robosuite/robosuite/models/assets/demonstrations/1638308941_8885477"
	hdf5_path = os.path.join(out_dir, "demo.hdf5")
	f = h5py.File(hdf5_path, "r")
	for demo_num in f['data'].keys():
		states = f['data'][demo_num]['states']
		actions = f['data'][demo_num]['actions']
		print('States: ',states.shape,' || Actions: ',actions.shape)
		env.sim.set_state_from_flattened(states[50])
		env.sim.forward()
		env.render()
		temp = env._get_observations()
		print(temp)
		time.sleep(2)
	

	f.close()

if __name__ == "__main__":
	config = load_controller_config(default_controller='OSC_POSE')
	env =suite.make(
			"Lift",
			robots="Sawyer",                # use Sawyer robot
			use_camera_obs=False,           # do not use pixel observations
			has_offscreen_renderer=False,   # not needed since not using pixel obs
			has_renderer=True,        # make sure we can render to the screen
			reward_shaping=True,            # use dense rewards
			control_freq=20,                # control should happen fast enough so that simulation looks smooth
			controller_configs = config,	# Controller config
		)

	gather_demonstrations_as_hdf5(env)
	
import numpy as np
import gym
import os, sys

import robosuite
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from ddpg_agent import ddpg_agent
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env,args):
    obs = env.reset()
    if args.env_name == 'robosuite':
        joint_c = obs['robot0_joint_pos_cos']
        joint_s = obs['robot0_joint_pos_sin']
        joint_angs = np.arctan2(joint_s,joint_c)
        state_space_size = joint_angs.shape[0]+obs['robot0_joint_vel'].shape[0]+obs['robot0_eef_pos'].shape[0]+obs['robot0_eef_quat'].shape[0]
        action_space_size = 7
        goal_space_size = 3
        if args.robosuite_string == 'TwoArmLift':
            joint_c = obs['robot1_joint_pos_cos']
            joint_s = obs['robot1_joint_pos_sin']
            joint_angs = np.arctan2(joint_s,joint_c)
            state_space_size = state_space_size+joint_angs.shape[0]+obs['robot1_joint_vel'].shape[0]+obs['robot1_eef_pos'].shape[0]+obs['robot1_eef_quat'].shape[0]
            action_space_size = action_space_size+7
            goal_space_size = goal_space_size+3

        params = {'obs': state_space_size,
                'goal': goal_space_size,
                'action': action_space_size,
                'action_max': 1.0,
                }
        params['max_timesteps'] = args.MAX_STEPS
        
    else:
        # close the environment
        params = {'obs': obs['observation'].shape[0],
                'goal': obs['desired_goal'].shape[0],
                'action': env.action_space.shape[0],
                'action_max': env.action_space.high[0],
                }
        params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the ddpg_agent
    if args.env_name == 'robosuite':
        config = load_controller_config(default_controller=args.CTRL_STRING)
        if args.robosuite_string == 'Lift':
            env =suite.make(
                args.robosuite_string,
                robots="Sawyer",                # use Sawyer robot
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=args.RENDER_ENV,        # make sure we can render to the screen
                reward_shaping=True,            # use dense rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
                controller_configs = config,	# Controller config
            )
        else:
            env =suite.make(
                args.robosuite_string,
                robots=["Sawyer","Sawyer"],                # use Sawyer robot
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=args.RENDER_ENV,        # make sure we can render to the screen
                reward_shaping=True,            # use dense rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
                controller_configs = config,	# Controller config
            )
    else:
        env = gym.make(args.env_name)
    
    # get the environment parameters
    env_params = get_env_params(env,args)

    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)

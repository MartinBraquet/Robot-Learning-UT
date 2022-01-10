import torch
from models import actor
from arguments import get_args
from train import get_env_params
import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    if args.env_name == 'robosuite':
        model_path = args.save_dir + args.env_name + '/'+args.robosuite_string +'/model.pt'
    else:
        model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, actor_model, _ = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    
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

    # get the env param
    observation = env.reset()
    env_params = get_env_params(env,args)
    
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(actor_model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        if args.env_name == 'robosuite':
            joint_c = observation['robot0_joint_pos_cos']
            joint_s = observation['robot0_joint_pos_sin']
            joint_angs = np.arctan2(joint_s,joint_c)
            obs = np.concatenate((joint_angs,observation['robot0_joint_vel'],observation['robot0_eef_pos'],observation['robot0_eef_quat']))
            g = observation['cube_pos']
            if args.robosuite_string == 'TwoArmLift':
                joint_c = observation['robot1_joint_pos_cos']
                joint_s = observation['robot1_joint_pos_sin']
                joint_angs = np.arctan2(joint_s,joint_c)
                obs = np.concatenate((obs,joint_angs,observation['robot1_joint_vel'],observation['robot1_eef_pos'],observation['robot1_eef_quat']))
                g = np.concatenate((observation['handle0_xpos'],observation['handle1_xpos']))
            else:
                g = observation['cube_pos']
        else:
            obs = observation['observation']
            g = observation['desired_goal']
        
        for t in range(env_params['max_timesteps']):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, done, info = env.step(action)
            if args.env_name == 'robosuite':
                joint_c = observation_new['robot0_joint_pos_cos']
                joint_s = observation_new['robot0_joint_pos_sin']
                joint_angs = np.arctan2(joint_s,joint_c)
                obs = np.concatenate((joint_angs,observation_new['robot0_joint_vel'],observation_new['robot0_eef_pos'],observation_new['robot0_eef_quat']))
                if args.robosuite_string == 'TwoArmLift':
                    joint_c = observation['robot1_joint_pos_cos']
                    joint_s = observation['robot1_joint_pos_sin']
                    joint_angs = np.arctan2(joint_s,joint_c)
                    obs = np.concatenate((obs,joint_angs,observation_new['robot1_joint_vel'],observation_new['robot1_eef_pos'],observation_new['robot1_eef_quat']))
            else: 
                obs = observation_new['observation']
            if done:
                break
        if args.env_name == 'robosuite':
            print('the episode is: {}, is success: {}'.format(i,env._check_success()))
        else:
            print('the episode is: {}, is success: {}'.format(i, info['is_success']))

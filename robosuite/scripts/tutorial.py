import numpy as np
from ray.rllib.agents import trainer
import robosuite as suite
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer
from robosuite.wrappers import GymWrapper
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# mujoco_robot = Panda()
# gripper = gripper_factory('PandaGripper')
# mujoco_robot.add_gripper(gripper)
# world = MujocoWorldBase()
# 
# mujoco_robot.set_base_xpos([0, 0, 0])
# world.merge(mujoco_robot)
# 
# mujoco_arena = TableArena()
# mujoco_arena.set_origin([0.8, 0, 0])
# world.merge(mujoco_arena)
# 
# sphere = BallObject(
#     name="sphere",
#     size=[0.04],
#     rgba=[0, 0.5, 0.5, 1]).get_obj()
# sphere.set('pos', '1.0 0 1.0')
# world.worldbody.append(sphere)
# 
# model = world.get_model(mode="mujoco_py")

# create environment instance
BallObject
env = GymWrapper(
        suite.make(
            "Lift",
            robots="Sawyer",                # use Sawyer robot
            use_camera_obs=False,           # do not use pixel observations
            has_offscreen_renderer=False,   # not needed since not using pixel obs
            has_renderer=False,              # make sure we can render to the screen
            reward_shaping=True,            # use dense rewards
            control_freq=20,                # control should happen fast enough so that simulation looks smooth
        )
    )
print(env.type)
# reset the environment
env.reset()

#sim = MjSim(model)
#viewer = MjViewer(sim)
#viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

ray.init()

trainer = PPOTrainer(env = env)
#tune.run(PPOTrainer)

#for i in range(10000):
#  sim.data.ctrl[:] = 0
#  sim.step()
#  viewer.render()
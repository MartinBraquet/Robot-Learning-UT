# Test the Two-Robot Environment (to include in our project)

import numpy as np
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda, Sawyer
from robosuite.models.grippers import gripper_factory

from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.utils.mjcf_utils import new_joint

from robosuite.models.arenas import EmptyArena, TableArena

from mujoco_py import MjSim, MjViewer

from TwoRobotEnvironment import buildTwoRobotEnvi

model = buildTwoRobotEnvi()

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(100000000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()


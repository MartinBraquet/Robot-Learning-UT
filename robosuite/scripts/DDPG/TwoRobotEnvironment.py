# New environment for the synergetic application: 2 robots lifting a bar

import numpy as np
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda, Sawyer
from robosuite.models.grippers import gripper_factory

from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.utils.mjcf_utils import new_joint

from robosuite.models.arenas import EmptyArena, TableArena

def buildTwoRobotEnvi():

    world = MujocoWorldBase()

    mujoco_robot = Panda()
    gripper = gripper_factory('PandaGripper')
    mujoco_robot.add_gripper(gripper)
    mujoco_robot.set_base_xpos([-0.7, 0, 0.6])
    world.merge(mujoco_robot)

    mujoco_robot2 = Panda(idn="robot2")
    mujoco_robot2.set_base_xpos([0.7, 0, 0.6])
    world.merge(mujoco_robot2)

    mujoco_arena = TableArena()
    mujoco_arena.set_origin([0, 0, 0])
    world.merge(mujoco_arena)

    sphere = BallObject(
        name="sphere",
        size=[0.04],
        rgba=[0, 0.5, 0.5, 1]).get_obj()
    sphere.set('pos', '1.5 0.0 2.0')
    world.worldbody.append(sphere)


    box = BoxObject(
        name="box",
        size=[0.4,0.4,1.0],
        rgba=[0, 0.5, 0.5, 1]).get_obj()
    box.set('pos', '0 1.0 1.0')
    world.worldbody.append(box)


    cyl = CylinderObject(
        name="cyl",
        size=[0.6,0.8],
        rgba=[0, 0.5, 0.5, 1]).get_obj()
    cyl.set('pos', '0.8 -0.8 0.85')
    world.worldbody.append(cyl)


    model = world.get_model(mode="mujoco_py")
    
    return model

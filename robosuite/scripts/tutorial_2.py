import numpy as np
import robosuite as suite
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from robosuite import load_controller_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, outputs):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_features = input_dim, out_features = hidden_dim, dtype=torch.float64)
        self.layer2 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim, dtype=torch.float64)
        self.layer3 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim, dtype=torch.float64)
        self.head = nn.Linear(in_features = hidden_dim, out_features = outputs, dtype=torch.float64)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.layer1(x)
        x = F.relu(x)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.head(x)

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            temp = policy_net(state)
            return temp
    else:
        temp = np.random.rand(1,n_actions)
        res = torch.tensor(temp, device=device)
        return res

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

if __name__ == "__main__":
    # create environment instance
    config = load_controller_config(default_controller="JOINT_POSITION")
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )

    # reset the environment
    obs = env.reset()

    print('Action Space Size: ',env.robots[0].dof)
    ref = np.array([1.0602402,   0.18473579,  0.01308697, -2.64412943,  0.01461617,  2.94620698, 0.80754049])

    n_actions = env.robots[0].dof

    policy_net = DQN(10, 64, n_actions).to(device)
    target_net = DQN(10, 64, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)


    steps_done = 0

    num_episodes = 50
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs = env.reset()
        joint_c = obs['robot0_joint_pos_cos']
        joint_s = obs['robot0_joint_pos_sin']
        #print('Joint Cosines, ', joint_c)
        #print('Joint Sines, ', joint_s)
        joint_angs = np.arctan2(joint_s,joint_c)
        state = np.concatenate(([obs['cube_pos']], [joint_angs]),axis=1,dtype=np.double)
        state = torch.from_numpy(state)
        print('State: ',state)
        for t in count():
            # Select and perform an action
            action = select_action(state)
            print('Action :',action.numpy()[0])
            _, reward, done, _ = env.step(action.numpy()[0])
            reward = torch.tensor([reward], device=device, dtype=torch.double)

            # Observe new state
            joint_c = obs['robot0_joint_pos_cos']
            joint_s = obs['robot0_joint_pos_sin']
            #print('Joint Cosines, ', joint_c)
            #print('Joint Sines, ', joint_s)
            joint_angs = np.arctan2(joint_s,joint_c)
            new_state = np.concatenate(([obs['cube_pos']], [joint_angs]),axis=1,dtype=np.double)
            if not done:
                next_state = torch.from_numpy(new_state)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                break
    
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


    for i in range(1000):
          # take action in the environment
        joint_c = obs['robot0_joint_pos_cos']
        joint_s = obs['robot0_joint_pos_sin']
        #print('Joint Cosines, ', joint_c)
        #print('Joint Sines, ', joint_s)
        joint_angs = np.arctan2(joint_s,joint_c)
        print('Joint angles', joint_angs)
        action = np.concatenate((ref-joint_angs,[0])) # sample random action
        obs, reward, done, info = env.step(action)
        print('Action: ', action, ' Reward: ', reward)
        env.render()  # render on display
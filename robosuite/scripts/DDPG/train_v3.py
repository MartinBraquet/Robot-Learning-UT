from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
import model
from os import listdir
from os.path import isfile, join

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer_v2:

	def __init__(self, state_dim, state_lim, action_dim, action_lim, ram, num_agents=1):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.state_lim = state_lim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.num_agents = num_agents
		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		self.actor = model.Actor(self.state_dim*self.num_agents, self.action_dim*self.num_agents, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim*self.num_agents, self.action_lim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = model.Critic(self.state_dim*self.num_agents, self.action_dim*self.num_agents)
		self.target_critic = model.Critic(self.state_dim*self.num_agents, self.action_dim*self.num_agents)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

		# Estimator for individual actions of each robot
		for i in range(num_agents):
			self.estimator[i] = model.Estimator(self.state_dim,self.state_lim)
			self.target_estimator[i] = model.Estimator(self.state_dim,self.state_lim)
			self.estimator_optimizer[i] = torch.optim.Adam(self.estimator[i].parameters(),LEARNING_RATE)
			utils.hard_update(self.target_estimator[i], self.estimator[i])
		
		# Estimator for combined actions of all robots
		self.joint_estimator = model.Joint_Estimator(self.state_dim*num_agents,self.state_lim)
		self.target_joint_estimator = model.Joint_Estimator(self.state_dim*num_agents,self.state_lim)
		self.joint_estimator_optimizer = torch.optim.Adam(self.joint_estimator.parameters(),LEARNING_RATE)

		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		utils.hard_update(self.target_joint_estimator, self.joint_estimator)

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state))
		action = self.target_actor.forward(state).detach()
		return action.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state))
		action = self.actor.forward(state).detach()
		new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)

		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s2).detach()
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + GAMMA*next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		# --------------------- optimize estimator -------------------
		pred_s2 = self.estimator.forward(s1)
		loss_estimator = F.smooth_l1_loss(pred_s2, s2)
		self.estimator_optimizer.zero_grad()
		loss_estimator.backward()
		self.estimator_optimizer.step()

		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.soft_update(self.target_critic, self.critic, TAU)
		utils.soft_update(self.target_estimator, self.estimator, TAU)

		# if self.iter % 100 == 0:
		# 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
		# 		' Loss_critic :- ', loss_critic.data.numpy()
		# self.iter += 1

	def save_models(self, episode_count, env_string):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		torch.save(self.target_actor.state_dict(), './Models/' + env_string + '/' + str(episode_count) + '_actor.pt')
		torch.save(self.target_critic.state_dict(), './Models/' + env_string + '/'+ str(episode_count) + '_critic.pt')
		torch.save(self.target_estimator.state_dict(), './Models/' + env_string + '/'+ str(episode_count) + '_estimator.pt')
		print('Models saved successfully: ',env_string,' ', episode_count)

	def load_models(self, episode, env_string):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		if episode == -1:
			onlyfiles = [f for f in listdir('./Models/'+env_string+'/') if isfile(join('./Models/'+env_string+'/', f))]
			max_val = 0
			print(onlyfiles)
			for fname in onlyfiles:
				arr = fname.split('_')
				if max_val < int(arr[0]):
					max_val = int(arr[0])
			episode = max_val
		self.actor.load_state_dict(torch.load('./Models/' + env_string + '/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + env_string + '/' + str(episode) + '_critic.pt'))
		self.estimator.load_state_dict(torch.load('./Models/' + env_string + '/' + str(episode) + '_estimator.pt'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		utils.hard_update(self.target_estimator, self.estimator)
		print('Models loaded succesfully: ',env_string,' ', episode)
		return episode
import argparse
import math

import gymnasium as gym
import pyglet
from pyglet.window import key

import miniworld
from tqdm import tqdm

import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
from miniworld.wrappers import PyTorchObsWrapper,GreyscaleWrapper

from rl_agent import Policy,select_action

train_args = dict(
    min_section_length=5,
    max_section_length=15,
    max_episode_steps=250,
    facing_forward=False,
    wall_tex='stripe_gradient'
)

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()
gamma = 1

def finish_episode():
    R = 0
    policy_loss = []
    value_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)
    # returns = (returns - returns.mean()) / (returns.std() + eps) # this create error ...
    for (log_prob,value), R in zip(policy.saved_log_probs, returns):
        advantage = R - value.item()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    policy.batch_loss.append(loss)
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def finish_batch():
    optimizer.zero_grad()
    loss = torch.stack(policy.batch_loss).sum()
    loss.backward()
    optimizer.step()
    del policy.batch_loss[:]
    return loss.item()

def train(nb_episode=3000):

    env = gym.make('MiniWorld-TaskHallwaySimple-v0', view="agent", render_mode=None,**train_args)
    env = PyTorchObsWrapper(env)
    
    running_reward = 0
    for i_episode in range(nb_episode):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 100):  # Don't infinite loop while learning
            action,show_prob = select_action(policy,state,training=True)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % 10 == 0:
            loss = finish_batch()
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward,loss),show_prob.cpu().detach().numpy()[0])
            env = gym.make('MiniWorld-TaskHallwaySimple-v0', view="agent", render_mode=None,**train_args)
            env = PyTorchObsWrapper(env)
        
train()

torch.save(policy.state_dict(), 'miniworld_agent_alt_texture.pt')
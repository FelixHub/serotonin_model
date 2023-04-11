
import gymnasium as gym
import miniworld
from miniworld.wrappers import PyTorchObsWrapper,GreyscaleWrapper
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args = dict(
    nb_sections=5,
    proba_change_motor_gain=1,
    motor_gains=[0.5,1,2],
    min_section_length=3,
    max_section_length=6,
    training=False,
    max_episode_steps=200,
)

'''
args = dict(
    nb_sections=2,
    proba_change_motor_gain=0,
    min_section_length=5,
    max_section_length=10,
    training=False,
    max_episode_steps=250,
)
'''


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.hidden_size = 32

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2).to(device)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2).to(device)
        self.maxpool = nn.MaxPool2d(2).to(device)

        self.outconvsize = 2016
        self.affine1 = nn.Linear( self.outconvsize , self.hidden_size).to(device) 

        ### replacement for rnn 
        # self.rnn = nn.RNN(self.outconvsize, self.hidden_size,batch_first=True).to(device)
        # self.hidden_state = torch.zeros(1, 1, self.hidden_size).to(device)

        self.action_head = nn.Sequential(
                                    nn.Linear(self.hidden_size, self.hidden_size).to(device),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, 3).to(device)
                                ).to(device)

        self.value_head = nn.Sequential(
                                    nn.Linear(self.hidden_size, self.hidden_size).to(device),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, 1).to(device)
                                ).to(device)

        self.saved_log_probs = []
        self.rewards = []
        self.batch_loss = []

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    # def reset_hidden_state(self):
    #     self.hidden_state = torch.zeros(1, 1, self.hidden_size).to(device)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = x.reshape(-1,self.outconvsize)
        out = self.relu(self.affine1(x))
        # h = self.rnn(x.unsqueeze(0), self.hidden_state)[1]
        # self.hidden_state = h
        # out = h.squeeze(0)
        action_prob = F.softmax(self.action_head(out), dim=-1)
        state_value = self.value_head(out)
        return action_prob, state_value

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs,state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append( (m.log_prob(action), state_value) )

    return action.item(),probs

policy = Policy().to(device)
policy.load_state_dict(torch.load('saved_models/miniworld_task.pt'))


def run_trajectories(nb_trajectories,id_run):

    env = gym.make('MiniWorld-TaskHallway-v0', 
                view="agent", render_mode=None,
                **args)

    env = PyTorchObsWrapper(env)

    trajectories = []
    trajectories_action = []

    nb_rewards = 0
    for i_trajectories in tqdm(range(nb_trajectories)):

        '''# we change environment every 10 trials => especially to capture gain changes

        if i_trajectories % 10 == 0 :
            env = gym.make('MiniWorld-TaskHallway-v0', 
                view="agent", render_mode=None,
                **args)
            env = PyTorchObsWrapper(env)
        '''

        observations = []
        actions = []
        observation, info = env.reset()

        for _ in range(args['max_episode_steps']):

            action,probs = select_action(observation) # agent policy that uses the observation and info

            # we save s_t and a_t
            actions.append(action)    
            observations.append(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
                if reward > 0 :
                    nb_rewards += 1

        observations = np.stack(observations)
        actions = np.stack(actions)
        
        trajectories.append(observations)
        trajectories_action.append(actions)

    print('proportion of rewarded trials :',nb_rewards/nb_trajectories)

    trajectories = np.stack(trajectories)
    trajectories_action = np.stack(trajectories_action)

    # convert to grayscale
    trajectories = np.sum(trajectories,axis=-3,keepdims=1)/3

    env.close()
    
    with open('data/rollout_changing_gain/agentRollout_observations_'+str(id_run)+'.npy', 'wb') as f:
        np.save(f, trajectories)
    with open('data/rollout_changing_gain/agentRollout_actions_'+str(id_run)+'.npy', 'wb') as f:
        np.save(f, trajectories_action)


for i_run in range(1):
    run_trajectories(nb_trajectories=1000,id_run=i_run)

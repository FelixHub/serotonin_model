# pylint: disable=E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.hidden_size = 32

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2).to(DEVICE)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2).to(DEVICE)
        self.maxpool = nn.MaxPool2d(2).to(DEVICE)

        self.outconvsize = 2016
        self.affine1 = nn.Linear(self.outconvsize, self.hidden_size).to(DEVICE)

        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size).to(DEVICE),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3).to(DEVICE),
        ).to(DEVICE)

        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size).to(DEVICE),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1).to(DEVICE),
        ).to(DEVICE)

        self.saved_log_probs = []
        self.rewards = []
        self.batch_loss = []

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = x.reshape(-1, self.outconvsize)
        out = self.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(out), dim=-1)
        state_value = self.value_head(out)
        return action_prob, state_value


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs, state_value = policy(state)
    probs_categorical = Categorical(probs)
    action = probs_categorical.sample()
    policy.saved_log_probs.append((probs_categorical.log_prob(action), state_value))

    return action.item(), probs

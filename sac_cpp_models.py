import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

STATE_DIM = 14  # 10 rangefinders + 2 velocity + 2 position
ACTION_DIM = 2  
HIDDEN_DIM = 256
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(HIDDEN_DIM, ACTION_DIM)
        self.log_std_layer = nn.Linear(HIDDEN_DIM, ACTION_DIM)
    
    def forward(self, x):
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.q_net(state_action)
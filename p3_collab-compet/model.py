import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1./np.sqrt(fan_in)
    return (-lim, lim)
    
class Actor(nn.Module):
    """ Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.seed = torch.manual_seed(seed)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(self.bn1(x)))
        return F.tanh(self.fc3(x))
    
    
class Critic(nn.Module):
    """ Critic model."""
    def __init__(self, state_sizes, action_sizes, seed, fcs1_units=128, fc2_units=256, fc3_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # self.bn1 = nn.BatchNorm1d(state_sizes+action_sizes)
        self.fcs1 = nn.Linear(state_sizes+action_sizes, fcs1_units)
        self.bn2 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        # xs = self.bn1(x)
        x = F.relu(self.fcs1(x))
        x = F.relu(self.bn2(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)        
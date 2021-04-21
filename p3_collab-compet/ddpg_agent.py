from collections import deque, namedtuple 
import random 
import copy
import torch 
from model import Actor, Critic 
from torch.optim import Adam 
import torch.nn.functional as F 
import numpy as np

BATCH_SIZE = 64
BUFFER_SIZE = int(10000)
WEIGHT_DECAY = 0.
ACTOR_LR = 5e-4
CRITIC_LR = 1e-3
UPDATE_EVERY = 4
GAMMA = 0.99
TAU = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples."""
    
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size 
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).astype(uint8).float().to(device)
        return (states, actions, rewards, next_states, dones) 
        
    def __len__(self):
        return len(self.memory)
    
    
class OUNoise:
    """ Ornstein-Uhlenbeck process. Adds noise to the deterministic action from the actor network. Contributes to exploration. Any other reason you would want such a noise added to the action ? """
    
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state 
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state 
    
        
class Agent:
    """ Interacts with and learns from the environment. """
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size 
        self.action_size = action_size 
        self.memory = ReplayBuffer(BATCH_SIZE, BUFFER_SIZE)
        
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr = ACTOR_LR)
        
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr = CRITIC_LR, weight_decay = WEIGHT_DECAY)
        
        self.noise = OUNoise(action_size)
        self.t_step = 0 
        
    def step(self, state, action, reward, next_state, done):
        e = self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step+1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences 
        
        # update critic 
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
        
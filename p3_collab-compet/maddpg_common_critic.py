import copy
import torch 
from model import Actor, Critic 
from torch.optim import Adam 
import torch.nn.functional as F 
import numpy as np

WEIGHT_DECAY = 0.
ACTOR_LR = 1e-4
CRITIC_LR = 5e-3
GAMMA = 0.995
TAU = 0.001
SEED = 0
# hyperparameters for the given tennis environment
STATE_SIZE = 24
ACTION_SIZE = 2 
NUM_AGENTS = 2 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
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

def decode(size, agent_id, arr):
    list_indices = torch.tensor([idx for idx in range(agent_id*size, agent_id*size+size)]).to(device)
    return arr.index_select(1, list_indices)
    
class Agent:
    """ Interacts with and learns from the environment. """
    # This declares a common critic for all the agents in the environment, and our agents hopefully learn faster with this change!
    critic_local = Critic(STATE_SIZE*NUM_AGENTS, ACTION_SIZE*NUM_AGENTS, 1).to(device)
    critic_target = Critic(STATE_SIZE*NUM_AGENTS, ACTION_SIZE*NUM_AGENTS, 1).to(device)
    critic_optimizer = Adam(critic_local.parameters(), lr=CRITIC_LR, weight_decay = WEIGHT_DECAY)
    def __init__(self, agent_id, num_agents, state_size, action_size):
        self.state_size = state_size 
        self.action_size = action_size 
        self.agent_id = agent_id
        self.actor_local = Actor(state_size, action_size, SEED).to(device)
        self.actor_target = Actor(state_size, action_size, SEED).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr = ACTOR_LR)
        
#         self.critic_local = Critic(state_size*num_agents, action_size*num_agents, SEED).to(device)
#         self.critic_target = Critic(state_size*num_agents, action_size*num_agents, SEED).to(device)
#         self.critic_optimizer = Adam(self.critic_local.parameters(), lr = CRITIC_LR, weight_decay = WEIGHT_DECAY)
        
        self.noise = OUNoise(action_size)
        self.t_step = 0 
        
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.unsqueeze(0)).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, other_agents, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences 

        # update critic
        own_states = decode(self.state_size, self.agent_id, states)
        other_states = decode(self.state_size, 1-self.agent_id, states)
        own_actions = decode(self.action_size, self.agent_id, actions)
        other_actions = decode(self.action_size, 1-self.agent_id, actions)
        own_next_states = decode(self.state_size, self.agent_id, next_states)
        other_next_states = decode(self.state_size, 1-self.agent_id, next_states)
        own_next_actions = self.actor_target(own_next_states)
        other_next_actions = other_agents[0].actor_target(other_next_states)

        # The common critic Q is designed to take states and actions in order, as in, Q(o_1, o_2, a_1, a_2), where o_1 is the observation of agent 1 and a_1 is the action taken by agent 1.
        if self.agent_id == 0:
            all_states = torch.cat((own_states, other_states), 1)
            all_actions = torch.cat((own_actions, other_actions), 1)
            all_next_states = torch.cat((own_next_states, other_next_states), 1)
            all_next_actions = torch.cat((own_next_actions, other_next_actions), 1)
        else:
            all_states = torch.cat((other_states, own_states), 1)
            all_actions = torch.cat((other_actions, own_actions), 1)
            all_next_states = torch.cat((other_next_states, own_next_states), 1)
            all_next_actions = torch.cat((other_next_actions, own_next_actions), 1)
            
        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        
        # extract the reward and done status of the current agent alone for Q updates
        r = torch.unsqueeze(rewards[:, self.agent_id], 1)
        do = torch.unsqueeze(dones[:, self.agent_id], 1)
        Q_targets = r + ((gamma*Q_targets_next)*(1-do))
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Use the agent's latest actor network to predict its own action. The other agents' action will not be predicted by the current agent, instead the other agents' action will remain the same as in the replay buffer 
        actions_pred_self = self.actor_local(own_states)
        actions_pred_other = other_actions # self.actor_local(other_states).detach()
        if self.agent_id == 0:
            actions_pred = torch.cat((actions_pred_self, actions_pred_other), 1)
        else:
            actions_pred = torch.cat((actions_pred_other, actions_pred_self), 1)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
        
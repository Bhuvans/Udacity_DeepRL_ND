from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

from collections import deque, namedtuple
from ddpg_agent import Agent
import torch
import random

BATCH_SIZE = 128
BUFFER_SIZE = int(1e6)
UPDATE_EVERY = 4
SEED = 1
UPDATES_PER_STEP = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples."""

    def __init__(self, batch_size, buffer_size, seed):
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([np.reshape(e.state, (1, -1)) for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([np.reshape(e.action, (1, -1)) for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([np.reshape(e.next_state, (1, -1)) for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def ddpg(num_episodes=10000, print_every=100):
    replay_memory = ReplayBuffer(BATCH_SIZE, BUFFER_SIZE, SEED)

    scores_deque = deque(maxlen=100)
    scores = []
    agents = []
    for agent_id in range(num_agents):
        agents.append(Agent(agent_id, num_agents, state_size, action_size))
    # Now on, let us assume the first component of states, rewards, done from the environment correspond to agent 0.

    for i_episode in range(num_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        for agent_id in range(num_agents):
            agents[agent_id].reset()
        score = np.zeros(num_agents)
        t = 0
        while True:
            actions = []
            with torch.no_grad():
                for agent_id in range(num_agents):
                    actions.append(agents[agent_id].act(states[agent_id, :]))
                actions = np.squeeze(np.array(actions), axis=1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            replay_memory.add(states, actions, rewards, next_states, dones)
            t = t + 1
            temp_t = t % UPDATE_EVERY
            if temp_t == 0:
                if len(replay_memory) > BATCH_SIZE:
                    for _ in range(UPDATES_PER_STEP):
                        for agent_id in range(num_agents):
                            experiences = replay_memory.sample()
                            other_agents = agents.copy()
                            other_agents.remove(agents[agent_id])
                            agents[agent_id].learn(experiences, other_agents)

            score += env_info.rewards
            states = next_states
            if np.any(dones):
                break
        scores.append(np.max(score))
        scores_deque.append(np.max(score))

        print('\rEpisode {} \t Score: {:.3f}, {:.3f}'.format(i_episode, score[0], score[1]), end='')
        if i_episode % print_every == 0:
            print('\r Episode {} \t Average Score: {:.3f} \t time steps/episode: {}'.format(i_episode, np.mean(scores_deque), t))
        if np.mean(scores_deque) >= 0.5:
            print('\r \t ****Task learned in {} episodes****'.format(i_episode))

    for i in range(num_agents):
        torch.save(agents[i].actor_local.state_dict(), 'checkpoint_actor_' + str(i) + '.pth')
        torch.save(agents[i].critic_local.state_dict(), 'checkpoint_critic_' + str(i) + '.pth')

    return scores


scores = ddpg()
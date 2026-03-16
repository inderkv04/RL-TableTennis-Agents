import numpy as np
import random
import copy
from collections import namedtuple, deque

from deep_rl.network.network import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6) 
BATCH_SIZE = 64        
GAMMA = 0.99            
TAU = 1e-3            
LR_ACTOR = 1e-4         
LR_CRITIC = 1e-3        
LEARN_EVERY = 1        
LEARN_NB = 4            
WEIGHT_DECAY = 0       

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent(object):

    def __init__(self, state_size, action_size, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.i_learn = 0 

        # Actor_Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        # Critic_Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,weight_decay=WEIGHT_DECAY)

        self.copy_weights(self.critic_local, self.critic_target)
        self.copy_weights(self.actor_local, self.actor_target)

        # Noise_process
        self.noise = OUNoise(2 * action_size, random_seed) # 2 is number of agents, here 2 tennis players

        # Replay_memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def copy_weights(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def step(self, states, actions, actions_other_player, rewards, next_states, next_states_other_player, dones):
        # FOR saving experience / reward for both agents(2 tennis agents)
        for state, action, action_other_player, reward, next_state, next_state_other_player,done in \
        zip(states, actions, actions_other_player, rewards, next_states, next_states_other_player, dones):
            self.memory.add(state, action, action_other_player, reward, next_state, next_state_other_player, done)

        self.i_learn = (self.i_learn + 1) % LEARN_EVERY
        # Learn every step if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.i_learn == 0:
            for _ in range(LEARN_NB):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, noise_factor=1.0):
       
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += noise_factor * self.noise.sample().reshape((-1, self.action_size))
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, actions_other_player, rewards, next_states, next_states_other_player, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions (for other player too) and Q values from target models
        actions_next = self.actor_target(next_states)
        actions_next_other_player = self.actor_target(next_states_other_player)
        Q_targets_next = self.critic_target(next_states, actions_next, actions_next_other_player)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions, actions_other_player)  # Current expected Q-values
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Clipping Policy updates
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred,actions_other_player).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "action_other_player", "reward","next_state", "next_state_other_player", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, action_other_player, reward, next_state, next_state_other_player, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, action_other_player, reward, next_state, next_state_other_player, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        actions_other_player = torch.from_numpy(np.vstack([e.action_other_player for e in experiences if e \
                                                           is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_states_other_player = torch.from_numpy(np.vstack([e.next_state_other_player for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, actions_other_player, rewards, next_states, next_states_other_player, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

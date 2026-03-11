#!/usr/bin/env python
# coding: utf-8

# # Deep Reinforcement learning - Training an Agent to Solve Unity-ML Tennis environment using MultiAgent DDPG algorithm
# ---
# 
# This notebook presents the code to train a Deep RL Agent to solve the Unity ML-Agent Tennis environment. The training uses Multi Agent Deep Deterministic Policy Gradient algorithm.
# 
# ### 1. Import the packages and Start the Environment
# 

# In[2]:


from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from deep_rl.agent.DDPG_agent import DDPGAgent


# In[3]:


# Tennis environment. In this environment, two agents control rackets to bounce a ball over a net.
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe") 


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
# 
# The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
# 
# Run the code cell below to print some information about the environment.

# In[4]:


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

print(brain.vector_observation_space_size)
print(state_size)
print(brain.vector_action_space_size)


# ### 3. Training

# In[ ]:


# create a new agent
agent = DDPGAgent(state_size=state_size, action_size=brain.vector_action_space_size, random_seed=1)


# In[6]:


#Just for debug/display purpose
print(state_size)


# In[7]:


#Just for debug/display purpose
actions = np.random.randn(num_agents, action_size)
print(actions)


# In[8]:


#Just for debug/display purpose
noise_factor = 0.1
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
print(states)
print(states.shape)
actions = agent.act(states, noise_factor=noise_factor)
print(actions)


# In[9]:


def train_maddpg(n_episodes=2000, max_t=1000, print_every=50):
    scores_deque = deque(maxlen=print_every)
    scores = []
    best_score = -np.inf
    noise_factor = 0.15  # A factor to multiply random noise
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations # states.shape is (2,24). env_reset returned state of both tennis players
        scores_agents = np.zeros(2)  # the scores of both tennis agents after an episode
        agent.reset()
        for t in range(max_t):
            
            # The Agent selects Actions
            # actions for both the agents (both tennis players)
            if i_episode < 20:
                actions = np.random.randn(num_agents, action_size)  # use random actions for the first 100 episodes
            else:
                actions = agent.act(states, noise_factor=noise_factor)
            
            # Actions of the other player
            actions_other_player = np.flip(actions, 0)
            
            # Environment processes the Action; produces new State, Rewards
            env_info = env.step(actions)[brain_name]      
            rewards = env_info.rewards                    
            next_states = env_info.vector_observations
            next_states_other_player = np.flip(next_states, 0)
            dones = env_info.local_done 
            
            # The Agent learns
            #agent.step(states, actions, rewards, next_states, dones)
            agent.step(states, actions, actions_other_player, rewards, next_states, next_states_other_player, dones) 
            
            states = next_states
            scores_agents += rewards
            if np.any(dones):
                break 
        avg_score = np.mean(scores_agents)  # the average score of the agents
        max_score = np.max(scores_agents)  # the max score of the agents
        #scores_deque.append(avg_score)
        scores_deque.append(max_score)
        #scores.append(avg_score)
        scores.append(max_score)
        
        #noise reduced during training as episodes progresses
        noise_factor = max(0.999 * noise_factor, 0.02)
        
        #print('\rEpisode {:d}\tscore: {:.2f}\taverage score over the last 10 episodes: {:.2f}'.format(i_episode, scores_deque[-1], np.mean(list(scores_deque)[-10:])), end="")
        print('\rEpisode {:d}\tmax score: {:.2f}\tavg max score over the last 10 episodes: {:.2f}'.format(i_episode, scores_deque[-1], np.mean(list(scores_deque)[-10:])), end="")
        #if i_episode % 10 == 0:
            #torch.save(agent.actor_local.state_dict(), 'weights/checkpoint_actor_{:d}_{:.2f}.pth'.format(i_episode, scores_deque[-1]))
            #torch.save(agent.critic_local.state_dict(), 'weights/checkpoint_critic_{:d}_{:.2f}.pth'.format(i_episode, scores_deque[-1]))
        
        if i_episode > 100 and np.mean(scores_deque) > 0.5 and np.mean(scores_deque) > best_score:
            best_score = np.mean(scores_deque)
            torch.save(agent.actor_local.state_dict(), 'weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'weights/checkpoint_critic.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
        
            break

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAvg Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores

scores = train_maddpg()


# In[4]:


#close the environment
env.close()


# ## 4. Plot Score vs Episode#

# In[5]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# ## 5. Performance Testing

# In[1]:


# Standalone script to test your trained agent
# This can be run after restarting the kernel

import os
import numpy as np
import torch
from unityagents import UnityEnvironment
from deep_rl.agent.DDPG_agent import DDPGAgent
import matplotlib.pyplot as plt
import random
import time

# Check if weights exist
if not os.path.exists('weights/checkpoint_actor.pth'):
    print("ERROR: Could not find trained weights at 'weights/checkpoint_actor.pth'")
    print("Please make sure you've trained the agent and saved the weights first.")
    exit()

# Use random worker_id to avoid connection conflicts
worker_id = random.randint(1, 99)
print(f"Creating Unity environment with worker_id={worker_id}...")

# Try to create environment with error handling
env = None
max_attempts = 5

for attempt in range(max_attempts):
    try:
        env = UnityEnvironment(
            file_name="Tennis_Windows_x86_64/Tennis.exe", 
            worker_id=worker_id + attempt  # Increment worker_id each attempt
        )
        print(f"✓ Environment created successfully on attempt {attempt + 1}")
        break
    except Exception as e:
        print(f"✗ Attempt {attempt + 1} failed: {str(e)[:100]}...")
        if attempt < max_attempts - 1:
            print(f"  Trying again with worker_id={worker_id + attempt + 1}...")
            time.sleep(2)
        else:
            print("\nFailed to create environment after all attempts.")
            print("Please:")
            print("1. Check Task Manager and end any 'Tennis.exe' processes")
            print("2. Wait a few seconds and try again")
            exit()

# Get environment information
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]

print(f"\nEnvironment Info:")
print(f"- Number of agents: {num_agents}")
print(f"- Action size: {action_size}")
print(f"- State size: {state_size}")

# Create agent and load trained weights
print("\nLoading trained agent...")
agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=1)

try:
    # Load the saved weights
    agent.actor_local.load_state_dict(torch.load('weights/checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('weights/checkpoint_critic.pth'))
    print("✓ Successfully loaded trained weights")
except Exception as e:
    print(f"✗ Error loading weights: {e}")
    env.close()
    exit()

# Put networks in evaluation mode
agent.actor_local.eval()
agent.critic_local.eval()

# Test the trained agent
n_test_episodes = 10
max_t = 1000
episode_scores = []
episode_lengths = []
agent_scores = []  # Store individual agent scores

print(f"\n{'='*60}")
print(f"Testing trained agent for {n_test_episodes} episodes")
print(f"{'='*60}\n")

for ep in range(1, n_test_episodes + 1):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    agent.reset()
    
    scores = np.zeros(num_agents)
    
    for t in range(1, max_t + 1):
        # Get actions from trained agent (no noise for testing)
        actions = agent.act(states, noise_factor=0.0)
        
        # Step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        
        scores += rewards
        states = next_states
        
        if np.any(dones):
            max_score = np.max(scores)
            print(f"Episode {ep:2d}: Max Score = {max_score:6.3f}, "
                  f"Agent Scores = [{scores[0]:6.3f}, {scores[1]:6.3f}], "
                  f"Length = {t:4d} steps")
            episode_scores.append(max_score)
            episode_lengths.append(t)
            agent_scores.append(scores.copy())
            break
    else:
        max_score = np.max(scores)
        print(f"Episode {ep:2d}: Max Score = {max_score:6.3f}, "
              f"Agent Scores = [{scores[0]:6.3f}, {scores[1]:6.3f}], "
              f"Length = {max_t:4d} steps (timeout)")
        episode_scores.append(max_score)
        episode_lengths.append(max_t)
        agent_scores.append(scores.copy())

# Calculate statistics
agent_scores = np.array(agent_scores)
print(f"\n{'='*60}")
print(f"PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"Max Score Statistics:")
print(f"  Mean:  {np.mean(episode_scores):6.3f} ± {np.std(episode_scores):6.3f}")
print(f"  Min:   {np.min(episode_scores):6.3f}")
print(f"  Max:   {np.max(episode_scores):6.3f}")
print(f"\nIndividual Agent Statistics:")
print(f"  Agent 1 Mean: {np.mean(agent_scores[:, 0]):6.3f} ± {np.std(agent_scores[:, 0]):6.3f}")
print(f"  Agent 2 Mean: {np.mean(agent_scores[:, 1]):6.3f} ± {np.std(agent_scores[:, 1]):6.3f}")
print(f"\nEpisode Statistics:")
print(f"  Episodes Solved (>0.5):  {np.sum(np.array(episode_scores) > 0.5)} / {n_test_episodes}")
print(f"  Mean Episode Length:     {np.mean(episode_lengths):.1f} steps")
print(f"  Success Rate:            {100 * np.sum(np.array(episode_scores) > 0.5) / n_test_episodes:.1f}%")
print(f"{'='*60}")

# Visualize results
fig = plt.figure(figsize=(15, 10))

# Plot 1: Episode scores
ax1 = plt.subplot(2, 2, 1)
ax1.plot(range(1, n_test_episodes + 1), episode_scores, 'b-', linewidth=2, marker='o', markersize=8)
ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Target Score (0.5)')
ax1.set_xlabel('Test Episode', fontsize=12)
ax1.set_ylabel('Max Score', fontsize=12)
ax1.set_title('Test Performance: Max Scores', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(bottom=-0.1)

# Plot 2: Individual agent scores
ax2 = plt.subplot(2, 2, 2)
ax2.plot(range(1, n_test_episodes + 1), agent_scores[:, 0], 'g-', linewidth=2, marker='s', label='Agent 1')
ax2.plot(range(1, n_test_episodes + 1), agent_scores[:, 1], 'm-', linewidth=2, marker='^', label='Agent 2')
ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Test Episode', fontsize=12)
ax2.set_ylabel('Individual Scores', fontsize=12)
ax2.set_title('Individual Agent Performance', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Episode lengths
ax3 = plt.subplot(2, 2, 3)
bars = ax3.bar(range(1, n_test_episodes + 1), episode_lengths, color='orange', alpha=0.7, edgecolor='darkorange')
ax3.set_xlabel('Test Episode', fontsize=12)
ax3.set_ylabel('Episode Length (steps)', fontsize=12)
ax3.set_title('Episode Lengths', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, length) in enumerate(zip(bars, episode_lengths)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{int(length)}', ha='center', va='bottom', fontsize=9)

# Plot 4: Score distribution
ax4 = plt.subplot(2, 2, 4)
ax4.hist(episode_scores, bins=10, color='skyblue', alpha=0.7, edgecolor='navy')
ax4.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Target Score')
ax4.axvline(x=np.mean(episode_scores), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(episode_scores):.3f}')
ax4.set_xlabel('Score', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Score Distribution', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()

plt.tight_layout()
plt.show()

# Clean up
print("\nClosing environment...")
env.close()
print("Testing complete!")

# Optional: Save test results
save_results = input("\nSave test results to file? (y/n): ").lower().strip() == 'y'
if save_results:
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Test Results - {timestamp}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Number of test episodes: {n_test_episodes}\n")
        f.write(f"Mean score: {np.mean(episode_scores):.3f} ± {np.std(episode_scores):.3f}\n")
        f.write(f"Min/Max scores: {np.min(episode_scores):.3f} / {np.max(episode_scores):.3f}\n")
        f.write(f"Success rate: {100 * np.sum(np.array(episode_scores) > 0.5) / n_test_episodes:.1f}%\n")
        f.write(f"\nDetailed Results:\n")
        for i, (score, length) in enumerate(zip(episode_scores, episode_lengths)):
            f.write(f"Episode {i+1}: Score={score:.3f}, Length={length}\n")
    
    print(f"Results saved to {filename}")


# In[1]:


# Complete solution that integrates with your existing code
# Just replace your train_maddpg function with this enhanced version

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from deep_rl.agent.DDPG_agent import DDPGAgent

def train_maddpg_enhanced(n_episodes=2000, max_t=1000, print_every=50):
    """Enhanced training with visualization"""
    
    # Standard tracking
    scores_deque = deque(maxlen=print_every)
    scores = []
    
    # Enhanced tracking for visualization
    agent1_scores = []
    agent2_scores = []
    episode_lengths = []
    
    best_score = -np.inf
    noise_factor = 0.15
    
    print("Starting Enhanced Multi-Agent Training...")
    print("="*60 + "\n")
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores_agents = np.zeros(2)
        agent.reset()
        
        for t in range(max_t):
            # The Agent selects Actions
            if i_episode < 20:
                actions = np.random.randn(num_agents, action_size)
            else:
                actions = agent.act(states, noise_factor=noise_factor)
            
            # Actions of the other player
            actions_other_player = np.flip(actions, 0)
            
            # Environment processes the Action; produces new State, Rewards
            env_info = env.step(actions)[brain_name]      
            rewards = env_info.rewards                    
            next_states = env_info.vector_observations
            next_states_other_player = np.flip(next_states, 0)
            dones = env_info.local_done 
            
            # The Agent learns
            agent.step(states, actions, actions_other_player, rewards, 
                      next_states, next_states_other_player, dones) 
            
            states = next_states
            scores_agents += rewards
            
            if np.any(dones):
                episode_lengths.append(t + 1)
                break 
        else:
            episode_lengths.append(max_t)
            
        # Track scores
        max_score = np.max(scores_agents)
        scores_deque.append(max_score)
        scores.append(max_score)
        agent1_scores.append(scores_agents[0])
        agent2_scores.append(scores_agents[1])
        
        # Reduce noise
        noise_factor = max(0.999 * noise_factor, 0.02)
        
        # Calculate metrics
        avg_score = np.mean(scores_deque)
        success_rate = sum(1 for s in scores[-100:] if s > 0.5) / min(100, len(scores)) * 100
        
        # Print progress
        print(f'\rEpisode {i_episode}\tMax: {max_score:.2f}\t'
              f'Avg: {avg_score:.2f}\tSuccess: {success_rate:.0f}%\t'
              f'Length: {episode_lengths[-1]}', end="")
        
        if i_episode % print_every == 0:
            print(f'\n\nEpisode {i_episode} Summary:')
            print(f'  Average Score: {avg_score:.2f}')
            print(f'  Success Rate: {success_rate:.1f}%')
            print(f'  Agent 1 Avg: {np.mean(agent1_scores[-print_every:]):.2f}')
            print(f'  Agent 2 Avg: {np.mean(agent2_scores[-print_every:]):.2f}')
            print(f'  Noise Factor: {noise_factor:.4f}\n')
        
        # Save best model
        if i_episode > 100 and avg_score > 0.5 and avg_score > best_score:
            best_score = avg_score
            torch.save(agent.actor_local.state_dict(), 'weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'weights/checkpoint_critic.pth')
            print(f'\n\n🎉 Environment solved in {i_episode-100} episodes!\t'
                  f'Average Score: {avg_score:.2f}\n')
            break
    
    print("\n" + "="*60)
    print("Training Complete! Creating visualizations...")
    print("="*60 + "\n")
    
    # Create comprehensive plots
    create_training_plots(scores, agent1_scores, agent2_scores, episode_lengths)
    
    return scores

def create_training_plots(scores, agent1_scores, agent2_scores, episode_lengths):
    """Create comprehensive training visualizations"""
    
    episodes = range(1, len(scores) + 1)
    
    # Create main figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Total Reward vs Episode (larger plot)
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(episodes, scores, 'b-', alpha=0.3, linewidth=1, label='Raw Scores')
    
    # Add multiple moving averages
    for window, color, style in [(10, 'lightblue', '-'), (50, 'blue', '-'), (100, 'darkblue', '-')]:
        if len(scores) >= window:
            ma = [np.mean(scores[max(0, i-window+1):i+1]) for i in range(len(scores))]
            ax1.plot(ma, color=color, linewidth=2, linestyle=style, label=f'{window}-Ep Avg')
    
    ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Target Score')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Max Score', fontsize=12)
    ax1.set_title('Total Reward vs Episode', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Success Rate
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    window = 100
    success_rates = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        rate = sum(1 for s in scores[start:i+1] if s > 0.5) / (i - start + 1) * 100
        success_rates.append(rate)
    
    ax2.plot(success_rates, 'purple', linewidth=2)
    ax2.fill_between(episodes, 0, success_rates, alpha=0.3, color='purple')
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate (100-Episode Window)')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    # 3. Episode Length
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.plot(episode_lengths, 'orange', alpha=0.5, linewidth=1)
    if len(episode_lengths) >= 50:
        ma = [np.mean(episode_lengths[max(0, i-49):i+1]) for i in range(len(episode_lengths))]
        ax3.plot(ma, 'darkred', linewidth=2, label='50-Ep Avg')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Individual Agent Performance
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax4.plot(agent1_scores, 'g-', alpha=0.5, linewidth=1, label='Agent 1')
    ax4.plot(agent2_scores, 'm-', alpha=0.5, linewidth=1, label='Agent 2')
    if len(agent1_scores) >= 50:
        ma1 = [np.mean(agent1_scores[max(0, i-49):i+1]) for i in range(len(agent1_scores))]
        ma2 = [np.mean(agent2_scores[max(0, i-49):i+1]) for i in range(len(agent2_scores))]
        ax4.plot(ma1, 'darkgreen', linewidth=2, label='Agent 1 Avg')
        ax4.plot(ma2, 'darkmagenta', linewidth=2, label='Agent 2 Avg')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Score')
    ax4.set_title('Individual Agent Scores')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary Statistics
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.axis('off')
    
    summary = f"""Training Summary:
    
Episodes: {len(scores)}
Best Score: {max(scores):.3f}
Average Score: {np.mean(scores):.3f}
Final 100-Ep Avg: {np.mean(scores[-100:]):.3f}

Success Rate: {sum(s > 0.5 for s in scores) / len(scores) * 100:.1f}%
Final Success Rate: {success_rates[-1]:.1f}%

Agent 1 Mean: {np.mean(agent1_scores):.3f}
Agent 2 Mean: {np.mean(agent2_scores):.3f}

Avg Episode Length: {np.mean(episode_lengths):.0f} steps
"""
    
    ax5.text(0.1, 0.9, summary, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Multi-Agent DDPG Training Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Run the enhanced training
if __name__ == "__main__":
    # Initialize environment (your existing code)
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    # Create agent
    agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=1)
    
    # Run enhanced training with visualization
    scores = train_maddpg_enhanced(n_episodes=2000, max_t=1000, print_every=50)
    
    # Close environment
    env.close()


# Above code Took 15 min to train 

# # 2nd Iteration (With better success criteria and visualization)

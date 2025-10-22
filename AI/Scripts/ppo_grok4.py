import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm


# Define the Actor network (policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output logits for Categorical distribution


# Define the Critic network (value function)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.lam = 0.95  # GAE lambda
        self.clip_param = 0.2  # Clipping parameter
        self.ppo_epochs = 10  # Number of optimization epochs
        self.minibatch_size = 64  # Minibatch size
        self.value_loss_coef = 0.5  # Value loss coefficient
        self.entropy_coef = 0.01  # Entropy coefficient
        self.max_grad_norm = 0.5  # Max gradient norm for clipping

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]  # Append the next value for bootstrapping
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - int(dones[i])) - values[i]
            gae = delta + self.gamma * self.lam * (1 - int(dones[i])) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        for _ in range(self.ppo_epochs):
            # Shuffle indices for minibatches
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Actor loss
                logits = self.actor(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic loss
                values = self.critic(mb_states).squeeze()
                critic_loss = self.value_loss_coef * (values - mb_returns).pow(2).mean()

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()


# Main training loop
def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim)

    num_updates = 1000  # Total updates
    rollout_size = 2048  # Steps per rollout
    max_ep_len = 500  # Max episode length for CartPole

    pbar = tqdm(range(num_updates))
    for update in pbar:
        # Collect rollout
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        ep_rewards = []  # For logging
        state, _ = env.reset()
        ep_reward = 0
        ep_len = 0

        while len(states) < rollout_size:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            state = next_state
            ep_reward += reward
            ep_len += 1

            if done or ep_len >= max_ep_len:
                ep_rewards.append(ep_reward)
                state, _ = env.reset()
                ep_reward = 0
                ep_len = 0

        # Get next value for last state
        next_value = agent.critic(torch.from_numpy(state).float().unsqueeze(0)).item()

        # Compute advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # Update agent
        agent.update(states, actions, log_probs, advantages, returns)

        # Update tqdm postfix with average reward
        avg_reward = np.mean(ep_rewards) if ep_rewards else 0
        pbar.set_postfix(avg_reward=f'{avg_reward:.2f}')

    env.close()


# Run training
if __name__ == "__main__":
    train()

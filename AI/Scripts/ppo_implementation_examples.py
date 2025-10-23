# ppo_minimal.py
# Minimal PPO from scratch for Gymnasium (PyTorch). Defaults to CartPole-v1.
# Discrete: Categorical policy. Continuous: Diagonal Gaussian policy.
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space, hidden=64):
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        act_dim = action_space.n if self.discrete else action_space.shape[0]

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

        if not self.discrete:
            # log std as a parameter vector (diagonal gaussian)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.body(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def get_dist(self, obs):
        logits, _ = self.forward(obs)
        if self.discrete:
            return Categorical(logits=logits)
        else:
            std = self.log_std.exp()
            return Normal(logits, std)

    def act(self, obs):
        with torch.no_grad():
            dist = self.get_dist(obs)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1) if not self.discrete else dist.log_prob(action)
        return action.cpu().numpy(), logp.cpu().numpy()

    def evaluate_actions(self, obs, actions):
        dist = self.get_dist(obs)
        if self.discrete:
            logp = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        _, v = self.forward(obs)
        return logp, entropy, v


def gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t + 1]
        delta = rewards[t] + gamma * values[t + 1] * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:-1]
    return adv, returns


def make_env(env_id="CartPole-v1", seed=0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    return env


def ppo_train(
        env_id="CartPole-v1", total_steps=200_000, rollout_steps=2048, update_epochs=10,
        minibatch_size=64, gamma=0.99, lam=0.95, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01,
        lr=3e-4, max_grad_norm=0.5, seed=0
):
    env = make_env(env_id, seed)
    obs_shape = env.observation_space.shape
    assert len(obs_shape) == 1, "This minimal script supports 1D observations."
    obs_dim = obs_shape[0]

    policy = ActorCritic(obs_dim, env.action_space).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    global_step = 0

    # Storage
    obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
    act_buf = []
    logp_buf = np.zeros((rollout_steps,), dtype=np.float32)
    rew_buf = np.zeros((rollout_steps,), dtype=np.float32)
    val_buf = np.zeros((rollout_steps + 1,), dtype=np.float32)
    done_buf = np.zeros((rollout_steps + 1,), dtype=np.float32)

    pbar = tqdm(range(total_steps // rollout_steps), desc="updates")
    for _ in pbar:
        # Collect rollout
        for t in range(rollout_steps):
            obs_buf[t] = obs
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                logits, v = policy.forward(obs_t)
                val_buf[t] = v.item()
                if policy.discrete:
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    logp = dist.log_prob(a)
                    action = a.item()
                else:
                    dist = Normal(logits, policy.log_std.exp())
                    a = dist.sample()
                    logp = dist.log_prob(a).sum(-1)
                    action = a.cpu().numpy()[0]
            act_buf.append(action)
            logp_buf[t] = logp.item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rew_buf[t] = reward
            done_buf[t + 1] = float(done)
            ep_ret += reward
            ep_len += 1
            obs = next_obs
            global_step += 1

            if done:
                obs, _ = env.reset()
                pbar.set_postfix({'episode_reward': ep_ret})
                ep_ret, ep_len = 0.0, 0

        # Bootstrap value
        with torch.no_grad():
            val_buf[-1] = policy.forward(torch.from_numpy(obs).float().to(device).unsqueeze(0))[1].item()

        # GAE + returns
        adv, ret = gae_advantages(rew_buf, val_buf, done_buf, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Prepare tensors
        obs_t = torch.from_numpy(obs_buf).float().to(device)
        if policy.discrete:
            acts_t = torch.tensor(act_buf, dtype=torch.long, device=device)
        else:
            acts_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        old_logp_t = torch.from_numpy(logp_buf).float().to(device)
        ret_t = torch.from_numpy(ret).float().to(device)
        adv_t = torch.from_numpy(adv).float().to(device)

        # PPO updates
        batch_size = rollout_steps
        idxs = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_acts = acts_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                new_logp, entropy, value = policy.evaluate_actions(mb_obs, mb_acts)

                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = 0.5 * ((mb_ret - value) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # clear buffers for next rollout
        act_buf.clear()

    env.close()


if __name__ == "__main__":
    # Example:
    # python ppo_minimal.py
    ppo_train(
        env_id="CartPole-v1",
        total_steps=100_000,
        rollout_steps=2048,
        update_epochs=10,
        minibatch_size=64,
        gamma=0.99, lam=0.95,
        clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01,
        lr=3e-4, seed=1
    )

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

import tensorflow as tf
import numpy as np
import gym

# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
gamma = 0.99  # Discount factor
lr_actor = 0.001  # Actor learning rate
lr_critic = 0.001  # Critic learning rate
clip_ratio = 0.2  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization


# Actor and Critic networks
class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.dense1(state)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value


# PPO algorithm
def ppo_loss(old_logits, old_values, advantages, states, actions, returns):
    def compute_loss(logits, values, actions, returns):
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)
        policy = tf.nn.softmax(logits)
        action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
        old_policy = tf.nn.softmax(old_logits)
        old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

        # Policy loss
        ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = tf.reduce_mean(tf.square(values - returns))

        # Entropy bonus (optional)
        entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus  # Entropy regularization
        return total_loss

    def get_advantages(returns, values):
        advantages = returns - values
        return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    def train_step(states, actions, returns, old_logits, old_values):
        with tf.GradientTape() as tape:
            logits, values = model(states)
            loss = compute_loss(logits, values, actions, returns)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    advantages = get_advantages(returns, old_values)
    for _ in range(epochs):
        loss = train_step(states, actions, returns, old_logits, old_values)
    return loss


# Initialize actor-critic model and optimizer
model = ActorCritic(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)

# Main training loop
max_episodes = 1000
max_steps_per_episode = 1000

for episode in range(max_episodes):
    states, actions, rewards, values, returns = [], [], [], [], []
    state = env.reset()
    for step in range(max_steps_per_episode):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        logits, value = model(state)

        # Sample action from the policy distribution
        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        state = next_state

        if done:
            returns_batch = []
            discounted_sum = 0
            for r in rewards[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns_batch.append(discounted_sum)
            returns_batch.reverse()

            states = tf.concat(states, axis=0)
            actions = np.array(actions, dtype=np.int32)
            values = tf.concat(values, axis=0)
            returns_batch = tf.convert_to_tensor(returns_batch)
            old_logits, _ = model(states)

            loss = ppo_loss(old_logits, values, returns_batch - np.array(values), states, actions, returns_batch)
            print(f"Episode: {episode + 1}, Loss: {loss.numpy()}")

            break

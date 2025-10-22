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

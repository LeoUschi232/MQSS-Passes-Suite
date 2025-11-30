# Gymnasium imports
from gymnasium.spaces import (
    Box, Dict, Discrete, Graph, GraphInstance, MultiBinary, MultiDiscrete, OneOf, Sequence, Text, Tuple
)
from gymnasium.wrappers.utils import RunningMeanStd
from gymnasium.error import CustomSpaceError
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.space import T_cov
from gymnasium import Space
import gymnasium as gym

# Standard library imports
from typing import Any, SupportsFloat
from collections.abc import Callable
from functools import singledispatch
from __future__ import annotations
import numpy as np


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float64):
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


class NormalizeReward(gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env[ObsType, ActType], gamma: float = 0.99, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.return_rms = RunningMeanStd(shape=())
        self.discounted_reward = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        self._update_running_mean = setting

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.discounted_reward = self.discounted_reward * self.gamma * (1 - terminated) + float(reward)
        if self._update_running_mean:
            self.return_rms.update(self.discounted_reward)
        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return obs, normalized_reward, terminated, truncated, info

"""
Multi-Agent Rollout Buffer for parameter-shared PPO.

Design:
  - Stores transitions for ALL agents simultaneously.
  - Since agents share parameters, each agent's transition is treated as an
    independent sample in a flat batch. A rollout of T steps with N agents
    produces T*N total training samples per update.
  - GAE (Generalized Advantage Estimation) is computed per-agent, then flattened.
"""

import torch
import numpy as np
from typing import List, Dict, Generator


class RolloutBuffer:
    """
    Stores rollout data for all agents and computes GAE advantages.

    Buffer shape conventions:
        [num_steps, num_agents] for scalars (rewards, values, dones, log_probs, actions)
        [num_steps, num_agents, feature_dim] for embeddings
    """

    def __init__(self, num_steps: int, num_agents: int, obs_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = device
        self._ptr = 0

        # Pre-allocate tensors on device
        self.observations = torch.zeros(num_steps, num_agents, obs_dim, device=device)
        self.actions     = torch.zeros(num_steps, num_agents, dtype=torch.long, device=device)
        self.log_probs   = torch.zeros(num_steps, num_agents, device=device)
        self.rewards     = torch.zeros(num_steps, num_agents, device=device)
        self.values      = torch.zeros(num_steps, num_agents, device=device)
        self.dones       = torch.zeros(num_steps, num_agents, device=device)

    def add(
        self,
        observations: torch.Tensor,  # [num_agents, obs_dim]
        actions: torch.Tensor,     # [num_agents]
        log_probs: torch.Tensor,   # [num_agents]
        rewards: torch.Tensor,     # [num_agents]
        values: torch.Tensor,      # [num_agents]
        dones: torch.Tensor,       # [num_agents]
    ) -> None:
        assert self._ptr < self.num_steps, "Buffer is full — call compute_returns() before adding more."
        self.observations[self._ptr] = observations
        self.actions[self._ptr]    = actions
        self.log_probs[self._ptr]  = log_probs
        self.rewards[self._ptr]    = rewards
        self.values[self._ptr]     = values
        self.dones[self._ptr]      = dones
        self._ptr += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,  # [num_agents] — critic estimate at last obs
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Computes GAE advantages and returns in-place.
        Stores results as self.advantages and self.returns (both [num_steps, num_agents]).
        """
        self.advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_agents, device=self.device)

        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, minibatch_size_in_steps: int) -> Generator:
        """
        Sample minibatches by TIMESTEP to preserve spatial graph structure.
        """
        # Normalize advantages globally first
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = torch.randperm(self.num_steps, device=self.device)
        for start in range(0, self.num_steps, minibatch_size_in_steps):
            mb_idx = indices[start:start + minibatch_size_in_steps]
            
            # mb_obs keeps [T, num_agents, obs_dim] structure for PyG batching
            mb_obs = self.observations[mb_idx]
            
            # Flatten the other tensors to [T * num_agents]
            mb_actions = self.actions[mb_idx].view(-1)
            mb_lp = self.log_probs[mb_idx].view(-1)
            mb_adv = adv[mb_idx].view(-1)
            mb_ret = self.returns[mb_idx].view(-1)
            
            yield (
                mb_obs,
                mb_actions,
                mb_lp,
                mb_adv,
                mb_ret,
            )

    def reset(self) -> None:
        self._ptr = 0

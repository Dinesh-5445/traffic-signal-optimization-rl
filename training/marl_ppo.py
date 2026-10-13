"""
MARL PPO Training Loop — CleanRL-style, parameter-shared.

Key design decisions:
  - GNNEncoder + PPOPolicy are trained jointly end-to-end.
  - All agents share weights — each agent provides independent training samples.
  - Standard PPO clipped objective with entropy regularization.
  - Separate optimizer for GNN encoder and policy heads (allows different LRs in future).
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, Data
from typing import List, Dict, Optional

from agents.gnn_encoder import GNNEncoder
from agents.ppo_policy import ParameterSharedPPOPolicy
from training.rollout_buffer import RolloutBuffer


class MARLTrainer:
    """
    Manages the full training lifecycle for parameter-shared MARL PPO.
    
    Attributes:
        gnn:    GNNEncoder — spatial feature extractor.
        policy: ParameterSharedPPOPolicy — shared Actor + Critic heads.
    """

    def __init__(
        self,
        env,
        ts_ids: List[str],
        graph_builder,
        device: torch.device,
        # Architecture
        node_feature_dim: int = 9,
        gnn_hidden: int = 64,
        gnn_out: int = 64,
        policy_hidden: int = 64,
        # PPO Hyperparameters
        num_steps: int = 128,
        num_epochs: int = 10,
        num_minibatches: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lr: float = 3e-4,
        seed: int = 42,
    ):
        self.env = env
        self.ts_ids = ts_ids
        self.graph_builder = graph_builder
        self.device = device
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_agents = len(ts_ids)

        # Seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Models
        self.gnn = GNNEncoder(
            in_channels=node_feature_dim,
            hidden_channels=gnn_hidden,
            out_channels=gnn_out,
        ).to(device)

        self.policy = ParameterSharedPPOPolicy(
            gnn_out_channels=gnn_out,
            action_dim=2,
            hidden_dim=policy_hidden,
        ).to(device)

        # Single optimizer for GNN + Policy (joint training)
        self.optimizer = optim.Adam(
            list(self.gnn.parameters()) + list(self.policy.parameters()),
            lr=lr,
            eps=1e-5,
        )

        self.buffer = RolloutBuffer(
            num_steps=num_steps,
            num_agents=self.num_agents,
            obs_dim=node_feature_dim,
            device=device,
        )

        # Minibatch size is now in terms of timesteps, not flat agent transitions
        self.minibatch_steps = num_steps // num_minibatches

    def _obs_to_graph(self, obs: Dict[str, np.ndarray]):
        """Converts dict of observations to a PyG graph and returns GNN embeddings."""
        graph = self.graph_builder.build_graph(env_state=None, obs=obs)
        graph = graph.to(self.device)
        embeddings = self.gnn(graph)  # [num_agents, gnn_out]
        return embeddings

    def _collect_rollout(self) -> Optional[Dict]:
        """Runs num_steps in the environment and fills the rollout buffer."""
        obs, infos = self.env.reset()
        self.buffer.reset()

        ep_rewards = []
        step_count = 0

        for step in range(self.num_steps):
            if not self.env.agents:
                break

            with torch.no_grad():
                embeddings = self._obs_to_graph(obs)
                actions_t, log_probs, _, values = self.policy.get_action_and_value(embeddings)

            actions_np = actions_t.cpu().numpy()
            action_dict = {ts: int(actions_np[i]) for i, ts in enumerate(self.ts_ids)}

            next_obs, rewards, terminations, truncations, infos = self.env.step(action_dict)

            reward_vec = torch.tensor(
                [rewards.get(ts, 0.0) for ts in self.ts_ids], dtype=torch.float32, device=self.device
            )
            done_vec = torch.tensor(
                [float(terminations.get(ts, False) or truncations.get(ts, False)) for ts in self.ts_ids],
                dtype=torch.float32, device=self.device,
            )

            # Convert obs dict to tensor to store in buffer
            obs_features = []
            for ts in self.ts_ids:
                if ts in obs:
                    obs_features.append(obs[ts])
                else:
                    obs_features.append(np.zeros_like(next(iter(obs.values()))))
            obs_tensor = torch.tensor(np.stack(obs_features), dtype=torch.float32, device=self.device)

            self.buffer.add(
                observations=obs_tensor,
                actions=actions_t,
                log_probs=log_probs.detach(),
                rewards=reward_vec,
                values=values.detach(),
                dones=done_vec,
            )

            ep_rewards.append(reward_vec.mean().item())
            obs = next_obs
            step_count += 1

        # Bootstrap last value
        if self.env.agents and obs:
            with torch.no_grad():
                last_embeddings = self._obs_to_graph(obs)
                _, _, _, last_values = self.policy.get_action_and_value(last_embeddings)
        else:
            last_values = torch.zeros(self.num_agents, device=self.device)

        self.buffer.compute_returns_and_advantages(last_values, self.gamma, self.gae_lambda)

        return {
            "mean_reward": np.mean(ep_rewards) if ep_rewards else 0.0,
            "steps_collected": step_count,
        }

    def _ppo_update(self) -> Dict[str, float]:
        """Runs num_epochs of PPO gradient updates over the collected buffer."""
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_updates = 0

        for _ in range(self.num_epochs):
            for batch in self.buffer.get_batches(self.minibatch_steps):
                mb_obs, mb_actions, mb_old_lp, mb_adv, mb_ret = batch

                # mb_obs has shape [minibatch_steps, num_agents, obs_dim]
                # Construct PyG graphs for the minibatch
                graph_list = []
                for t in range(mb_obs.size(0)):
                    g = Data(x=mb_obs[t], edge_index=self.graph_builder.edge_index.to(self.device))
                    graph_list.append(g)
                
                batched_graph = Batch.from_data_list(graph_list)
                
                # Forward pass through GNN to get gradients
                mb_embed = self.gnn(batched_graph)

                _, new_lp, entropy, new_values = self.policy.get_action_and_value(mb_embed, mb_actions)

                # PPO clipped surrogate objective
                ratio = torch.exp(new_lp - mb_old_lp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (unclipped — simpler, CleanRL default)
                vf_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()

                # Total loss
                loss = pg_loss - self.ent_coef * entropy + self.vf_coef * vf_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.gnn.parameters()) + list(self.policy.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()
                total_updates += 1

        n = max(total_updates, 1)
        return {
            "policy_loss": total_pg_loss / n,
            "value_loss":  total_vf_loss / n,
            "entropy":     total_entropy / n,
        }

    def train(self, total_timesteps: int, save_dir: str = "results/v2") -> None:
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        update = 0
        global_step = 0
        start_time = time.time()

        print(f"{'Update':>8} {'Steps':>10} {'MeanReward':>14} {'PG Loss':>12} {'VF Loss':>12} {'Entropy':>10}")
        print("-" * 70)

        while global_step < total_timesteps:
            rollout_stats = self._collect_rollout()
            if rollout_stats is None:
                break

            update_stats = self._ppo_update()

            global_step += rollout_stats["steps_collected"] * self.num_agents
            update += 1

            print(
                f"{update:>8d} {global_step:>10d} "
                f"{rollout_stats['mean_reward']:>14.4f} "
                f"{update_stats['policy_loss']:>12.4f} "
                f"{update_stats['value_loss']:>12.4f} "
                f"{update_stats['entropy']:>10.4f}"
            )

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")

        # Save models
        torch.save(self.gnn.state_dict(), os.path.join(save_dir, "gnn_encoder.pt"))
        torch.save(self.policy.state_dict(), os.path.join(save_dir, "ppo_policy.pt"))
        print(f"Models saved to {save_dir}/")

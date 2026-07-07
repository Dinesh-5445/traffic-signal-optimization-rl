"""
metrics.py
----------
Shared metric extraction and aggregation utilities for Phase 4 & 5 benchmarking.

Metric Definitions
------------------
- queue_length_per_step : total halted vehicles across all controlled incoming
  edges at the END of each environment step.
- episode_total_reward  : sum of reward over all steps in one episode.
- avg_queue_per_step    : episode_total_reward / episode_steps (negative of this
  gives average queue length since reward = -queue).
- throughput            : total vehicles that arrived (finished route) during the episode.
- waiting_time_per_step : sum of waiting times of all vehicles on controlled edges per step.
- signal_switches       : number of times a traffic signal phase changed during the episode.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EpisodeMetrics:
    """Holds raw metric data for a single episode."""

    def __init__(self, policy_name: str, episode_idx: int, seed: Optional[int] = None) -> None:
        self.policy_name = policy_name
        self.episode_idx = episode_idx
        self.seed = seed

        self.step_rewards: List[float] = []
        self.step_waiting_times: List[float] = []
        # per_intersection_queues[tl_id] = list of queue lengths per step
        self.per_intersection_queues: Dict[str, List[float]] = {}
        self.terminated: bool = False
        self.truncated: bool = False
        self.throughput: int = 0
        self.signal_switches: int = 0
        self._last_action: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def record_step(self, reward: float, obs_flat: np.ndarray, ts_ids: List[str], waiting_time: float, action: np.ndarray) -> None:
        """Record one environment step."""
        self.step_rewards.append(float(reward))
        self.step_waiting_times.append(float(waiting_time))
        
        # Track signal switches
        if self._last_action is not None:
            # Action is shape (5,)
            switches = int(np.sum(action != self._last_action))
            self.signal_switches += switches
        self._last_action = np.copy(action)

        for i, tl in enumerate(ts_ids):
            base = i * 5
            q = float(obs_flat[base] + obs_flat[base + 1] +
                       obs_flat[base + 2] + obs_flat[base + 3])
            self.per_intersection_queues.setdefault(tl, []).append(q)

    # ------------------------------------------------------------------
    @property
    def total_reward(self) -> float:
        """Total reward accumulated during the episode."""
        return sum(self.step_rewards)

    @property
    def num_steps(self) -> int:
        """Total number of steps in the episode."""
        return len(self.step_rewards)

    @property
    def avg_queue_per_step(self) -> float:
        """Average total queue length across all steps (positive number)."""
        if not self.step_rewards:
            return 0.0
        return -self.total_reward / self.num_steps

    @property
    def var_queue(self) -> float:
        """Variance of total queue length."""
        if not self.step_rewards:
            return 0.0
        # step_rewards = -queue, so variance is same
        return float(np.var(self.step_rewards))

    @property
    def avg_waiting_time(self) -> float:
        """Average waiting time across all steps."""
        if not self.step_waiting_times:
            return 0.0
        return sum(self.step_waiting_times) / self.num_steps

    @property
    def avg_per_intersection(self) -> Dict[str, float]:
        """Average queue length per specific intersection."""
        return {
            tl: (sum(qs) / len(qs) if qs else 0.0)
            for tl, qs in self.per_intersection_queues.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for CSV exporting."""
        d = {
            "policy": self.policy_name,
            "episode": self.episode_idx,
            "seed": self.seed,
            "total_reward": round(self.total_reward, 2),
            "num_steps": self.num_steps,
            "avg_queue_per_step": round(self.avg_queue_per_step, 2),
            "var_queue": round(self.var_queue, 2),
            "avg_waiting_time": round(self.avg_waiting_time, 2),
            "throughput": self.throughput,
            "signal_switches": self.signal_switches,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        for tl, avg_q in self.avg_per_intersection.items():
            d[f"avg_queue_{tl}"] = round(avg_q, 2)
        return d


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_episodes(episodes: List[EpisodeMetrics]) -> Dict[str, Any]:
    """Compute mean ± stdev across a list of episodes for a single policy."""
    import statistics

    rewards = [e.total_reward for e in episodes]
    queues = [e.avg_queue_per_step for e in episodes]
    var_queues = [e.var_queue for e in episodes]
    waiting_times = [e.avg_waiting_time for e in episodes]
    throughputs = [e.throughput for e in episodes]
    switches = [e.signal_switches for e in episodes]

    def _stats(values: List[float]) -> Tuple[float, float]:
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return round(mean, 2), round(std, 2)

    reward_mean, reward_std = _stats(rewards)
    queue_mean, queue_std = _stats(queues)
    var_queue_mean, _ = _stats(var_queues)
    wait_mean, wait_std = _stats(waiting_times)
    thru_mean, thru_std = _stats(throughputs)
    sw_mean, sw_std = _stats(switches)

    return {
        "policy": episodes[0].policy_name,
        "num_episodes": len(episodes),
        "mean_total_reward": reward_mean,
        "std_total_reward": reward_std,
        "mean_avg_queue_per_step": queue_mean,
        "std_avg_queue_per_step": queue_std,
        "mean_var_queue": var_queue_mean,
        "mean_avg_waiting_time": wait_mean,
        "std_avg_waiting_time": wait_std,
        "mean_throughput": thru_mean,
        "std_throughput": thru_std,
        "mean_signal_switches": sw_mean,
        "std_signal_switches": sw_std
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_episodes_csv(episodes: List[EpisodeMetrics], output_path: str) -> None:
    """Save raw episode data to a CSV file."""
    if not episodes:
        return
    rows = [e.to_dict() for e in episodes]
    keys = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Episode CSV saved -> {output_path}")


def save_summary_json(summaries: List[Dict[str, Any]], output_path: str) -> None:
    """Save aggregated metric summaries to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Summary JSON saved -> {output_path}")


def print_comparison_table(summaries: List[Dict[str, Any]]) -> None:
    """Pretty-print a benchmark comparison table to stdout."""
    header = (
        f"{'Policy':<20} {'Episodes':>9} "
        f"{'Queue':>9} {'Q-Var':>9} "
        f"{'WaitTime':>9} {'Thruput':>9} "
        f"{'Switches':>9} {'Reward':>11}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("PHASE 5: PPO vs BASELINE COMPARISON")
    print(sep)
    print(header)
    print(sep)
    for s in summaries:
        print(
            f"{s['policy']:<20} {s['num_episodes']:>9} "
            f"{s['mean_avg_queue_per_step']:>9.1f} "
            f"{s['mean_var_queue']:>9.1f} "
            f"{s['mean_avg_waiting_time']:>9.1f} "
            f"{s['mean_throughput']:>9.1f} "
            f"{s['mean_signal_switches']:>9.1f} "
            f"{s['mean_total_reward']:>11.0f}"
        )
    print(sep + "\n")

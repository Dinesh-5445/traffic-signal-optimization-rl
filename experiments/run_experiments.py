"""
Experiment runner for the V2 experiment matrix (EXP-00 to EXP-05).

Each experiment is defined as a named configuration that controls:
  - Which architecture to use (Static / Independent MARL / Param-Shared / + GNN)
  - Which reward function to use (local queue / + co-op / multi-objective)
  - Training hyperparameters

Usage:
    python experiments/run_experiments.py --exp EXP-05 --sumo-cfg networks/grid.sumocfg
    python experiments/run_experiments.py --exp all    --sumo-cfg networks/grid.sumocfg

Results are saved to results/<exp_id>/ for comparison and plotting.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

import traci
import sumolib

from environment.builders import ObsBuilder, MultiObjectiveRewardBuilder
from environment.multi_agent_env import TrafficMARLEnv
from communication.graph_builder import GraphBuilder
from training.marl_ppo import MARLTrainer
from evaluation.metrics_collector import MetricsCollector, MetricsLogger, EpisodeMetrics

# ─────────────────────────────────────────────────────────────────────────────
# Shared network constants
# ─────────────────────────────────────────────────────────────────────────────
TS_IDS = ['A1', 'B0', 'B1', 'B2', 'C1']
TL_EDGES = {
    'A1': {'N': 'A2A1', 'S': 'A0A1', 'E': 'B1A1', 'W': None},
    'B0': {'N': 'B1B0', 'S': None,   'E': 'C0B0', 'W': 'A0B0'},
    'B1': {'N': 'B2B1', 'S': 'B0B1', 'E': 'C1B1', 'W': 'A1B1'},
    'B2': {'N': None,   'S': 'B1B2', 'E': 'C2B2', 'W': 'A2B2'},
    'C1': {'N': 'C2C1', 'S': 'C0C1', 'E': None,   'W': 'B1C1'},
}
NEIGHBOR_MAP = {
    'A1': ['B1'],
    'B0': ['B1'],
    'B1': ['A1', 'B0', 'B2', 'C1'],
    'B2': ['B1'],
    'C1': ['B1'],
}


# ─────────────────────────────────────────────────────────────────────────────
# EXP-00: Static Timing Baseline (no RL)
# ─────────────────────────────────────────────────────────────────────────────
def run_exp00_static(sumo_cfg: str, episodes: int = 5, max_steps: int = 500) -> None:
    """
    Run SUMO with fixed-timing program (no agent). Records baseline metrics.
    Each intersection cycles through its default fixed-time programme.
    """
    print("\n" + "=" * 60)
    print("  EXP-00: Static Fixed-Timing Baseline")
    print("=" * 60)

    results_dir = "results/EXP-00"
    os.makedirs(results_dir, exist_ok=True)
    logger = MetricsLogger(os.path.join(results_dir, "metrics.csv"))
    collector = MetricsCollector(TS_IDS)

    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "--no-step-log"]

    for ep in range(episodes):
        if traci.isLoaded():
            traci.close()
        traci.start(sumo_cmd)

        collector.reset()
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            # Static: no agent action, collect metrics manually
            rewards = {ts: 0.0 for ts in TS_IDS}  # placeholder
            collector.step(rewards)
            step += 1

        metrics = collector.summarise(episode=ep + 1)
        logger.log(metrics)
        traci.close()
        print(f"  Episode {ep+1}: Wait={metrics.mean_wait_time:.2f}s, Throughput={metrics.throughput}, Teleports={metrics.teleport_count}")

    logger.save()
    logger.print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build env and train MARL
# ─────────────────────────────────────────────────────────────────────────────
def _build_and_train(
    exp_id: str,
    sumo_cfg: str,
    use_gnn: bool = True,
    use_coop_reward: bool = True,
    total_timesteps: int = 100_000,
    seed: int = 42,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {exp_id}: use_gnn={use_gnn}, use_coop={use_coop_reward}")
    print("=" * 60)

    results_dir = f"results/{exp_id}"
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "--no-step-log"]

    obs_builder   = ObsBuilder(TS_IDS, TL_EDGES)

    # Co-op reward vs local-only reward
    neighbor_map  = NEIGHBOR_MAP if use_coop_reward else {ts: [] for ts in TS_IDS}
    reward_builder = MultiObjectiveRewardBuilder(TS_IDS, TL_EDGES, neighbor_map)
    graph_builder  = GraphBuilder(TS_IDS, TL_EDGES)

    env = TrafficMARLEnv(
        sumo_cmd=sumo_cmd,
        ts_ids=TS_IDS,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        graph_builder=graph_builder,
        max_steps=500,
    )

    trainer = MARLTrainer(
        env=env,
        ts_ids=TS_IDS,
        graph_builder=graph_builder,
        device=device,
        seed=seed,
    ) if use_gnn else MARLTrainer(
        env=env,
        ts_ids=TS_IDS,
        graph_builder=graph_builder,
        device=device,
        seed=seed,
        # Without GNN: node_feature_dim stays the same, but GNN still runs.
        # True "no-GNN" ablation would bypass GNN — this stub marks the intent.
        # Full no-GNN ablation requires a separate FlatMLP policy (future work).
    )

    trainer.train(total_timesteps=total_timesteps, save_dir=results_dir)
    env.close()
    print(f"  {exp_id} complete — models saved to {results_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment dispatch table
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "EXP-00": lambda cfg, steps: run_exp00_static(cfg, episodes=5),
    "EXP-02": lambda cfg, steps: _build_and_train("EXP-02", cfg, use_gnn=False, use_coop_reward=False, total_timesteps=steps),
    "EXP-03": lambda cfg, steps: _build_and_train("EXP-03", cfg, use_gnn=False, use_coop_reward=True,  total_timesteps=steps),
    "EXP-04": lambda cfg, steps: _build_and_train("EXP-04", cfg, use_gnn=False, use_coop_reward=True,  total_timesteps=steps),
    "EXP-05": lambda cfg, steps: _build_and_train("EXP-05", cfg, use_gnn=True,  use_coop_reward=True,  total_timesteps=steps),
    "EXP-06": lambda cfg, steps: _build_and_train("EXP-06", cfg, use_gnn=True,  use_coop_reward=True,  total_timesteps=steps),
}


def main():
    parser = argparse.ArgumentParser(description="Run V2 experiment matrix.")
    parser.add_argument("--exp",       default="EXP-05", choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run (or 'all' to run all).")
    parser.add_argument("--sumo-cfg",  required=True, help="Path to SUMO .sumocfg file.")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps for RL experiments.")
    args = parser.parse_args()

    if args.exp == "all":
        for name, fn in EXPERIMENTS.items():
            fn(args.sumo_cfg, args.timesteps)
    else:
        EXPERIMENTS[args.exp](args.sumo_cfg, args.timesteps)


if __name__ == "__main__":
    main()

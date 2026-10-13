"""
Final Demonstration Script for Phase 5 (Version 2)
Multi-Agent Traffic Signal Optimization

This script loads the finalized EXP-06 trained GNN encoder and PPO policy,
launches the SUMO GUI, and controls the traffic signals via TraCI using the
learned agent policies.

Usage:
    python scripts/demo_rl_gui.py

You can override the defaults:
    python scripts/demo_rl_gui.py --sumo-cfg networks/grid_rush_hour.sumocfg
"""

import argparse
import os
import sys
import time

# Use TraCI for sumo-gui. While libsumo is faster, it doesn't support GUI directly unless built with it.
# To be safe for GUI, we disable libsumo for this specific demo script.
os.environ.pop("LIBSUMO_AS_TRACI", None)
import traci

import torch
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gnn_encoder import GNNEncoder
from agents.ppo_policy import ParameterSharedPPOPolicy
from environment.builders import ObsBuilder, MultiObjectiveRewardBuilder
from environment.multi_agent_env import TrafficMARLEnv
from communication.graph_builder import GraphBuilder

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Network topology
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

NODE_FEAT_DIM  = 9
GNN_HIDDEN     = 64
GNN_OUT        = 64
POLICY_HIDDEN  = 64


def main():
    parser = argparse.ArgumentParser(description="Final RL GUI Demonstration")
    parser.add_argument("--gnn-path",    default="results/EXP-06/gnn_encoder.pt", help="Path to saved GNN encoder")
    parser.add_argument("--policy-path", default="results/EXP-06/ppo_policy.pt", help="Path to saved PPO policy")
    parser.add_argument("--sumo-cfg",    default="networks/grid_balanced.sumocfg", help="Path to SUMO .sumocfg file")
    parser.add_argument("--sim-horizon", type=int, default=1200, help="Simulation duration in seconds")
    parser.add_argument("--delay",       type=float, default=0.0, help="Optional artificial delay to slow down GUI (e.g. 0.05)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Launching RL GUI Demonstration ---")
    print(f"Device:      {device}")
    print(f"Scenario:    {args.sumo_cfg}")
    print(f"Horizon:     {args.sim_horizon}s")

    # â”€â”€ Load models
    gnn = GNNEncoder(NODE_FEAT_DIM, GNN_HIDDEN, GNN_OUT).to(device)
    policy = ParameterSharedPPOPolicy(GNN_OUT, action_dim=2, hidden_dim=POLICY_HIDDEN).to(device)

    try:
        gnn.load_state_dict(torch.load(args.gnn_path, map_location=device, weights_only=True))
        policy.load_state_dict(torch.load(args.policy_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Could not find model files at {args.gnn_path} or {args.policy_path}.")
        print("Ensure EXP-06 was completed successfully.")
        sys.exit(1)

    gnn.eval()
    policy.eval()
    print("Models loaded successfully.")

    # â”€â”€ Build environment
    sumo_cmd = ["sumo-gui", "-c", args.sumo_cfg, "--no-warnings", "--no-step-log"]
    obs_builder   = ObsBuilder(TS_IDS, TL_EDGES)
    reward_builder = MultiObjectiveRewardBuilder(TS_IDS, TL_EDGES, NEIGHBOR_MAP)
    graph_builder  = GraphBuilder(TS_IDS, TL_EDGES)

    # Note: Using standard TraCI for GUI, delay logic might be required natively via SUMO.
    if args.delay > 0:
        sumo_cmd.extend(["--step-length", str(args.delay)])

    env = TrafficMARLEnv(
        sumo_cmd=sumo_cmd,
        ts_ids=TS_IDS,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        graph_builder=graph_builder,
        max_steps=int(args.sim_horizon / 5),
    )

    print("\nStarting simulation... (Switching to GUI, please ensure X11/Display is available if not on local desktop)")
    obs, _ = env.reset()

    # Tracking for verbose output
    last_phases = {tl: None for tl in TS_IDS}

    while env.agents:
        if not obs:
            break

        graph = graph_builder.build_graph(env_state=None, obs=obs).to(device)

        with torch.no_grad():
            embeddings = gnn(graph)
            actions_t, _, _, _ = policy.get_action_and_value(embeddings)

        actions_np = actions_t.cpu().numpy()
        action_dict = {ts: int(actions_np[i]) for i, ts in enumerate(TS_IDS)}

        # Execute step
        obs, rewards, terminations, truncations, _ = env.step(action_dict)

        sim_time = traci.simulation.getTime()

        # Log phase changes to terminal for evidence
        for tl in TS_IDS:
            current_phase = traci.trafficlight.getPhase(tl)
            if last_phases[tl] is not None and current_phase != last_phases[tl]:
                # Identify if it's a green phase
                phase_name = "N/S Green" if current_phase == 0 else ("E/W Green" if current_phase == 2 else "Yellow")
                print(f"Time {sim_time:>6.1f}s | {tl} switched to {phase_name} (Phase {current_phase}) | Action={action_dict[tl]}")
            last_phases[tl] = current_phase

        if args.delay > 0:
            time.sleep(args.delay)

    env.close()
    print("\nDemonstration complete.")

if __name__ == "__main__":
    main()

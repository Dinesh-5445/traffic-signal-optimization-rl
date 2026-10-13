import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
import traci

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gnn_encoder import GNNEncoder
from agents.ppo_policy import ParameterSharedPPOPolicy
from environment.builders import ObsBuilder, MultiObjectiveRewardBuilder
from environment.multi_agent_env import TrafficMARLEnv
from communication.graph_builder import GraphBuilder
from evaluation.metrics_collector import MetricsCollector, MetricsLogger

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

NODE_FEAT_DIM = 9
GNN_HIDDEN = 64
GNN_OUT = 64
POLICY_HIDDEN = 64

SCENARIOS = ["balanced", "ns_heavy", "ew_heavy", "rush_hour", "random", "congested"]
MODELS = {
    "Static": None,
    "V1_PPO": "results/EXP-02",
    "V2_MARL_GNN": "results/EXP-05",
}

def evaluate_static(sumo_cfg, episodes=1, max_steps=1200):
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "--no-step-log"]
    collector = MetricsCollector(TS_IDS)
    all_metrics = []

    for ep in range(episodes):
        if traci.isLoaded():
            traci.close()
        traci.start(sumo_cmd)
        collector.reset()
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            rewards = {ts: 0.0 for ts in TS_IDS}
            collector.step(rewards)
            step += 1
            
        metrics = collector.summarise(episode=ep + 1)
        all_metrics.append(metrics)
        traci.close()
        
    return all_metrics

def evaluate_rl(sumo_cfg, model_dir, device, episodes=1, max_steps=1200):
    gnn_path = os.path.join(model_dir, "gnn_encoder.pt")
    policy_path = os.path.join(model_dir, "ppo_policy.pt")
    
    gnn = GNNEncoder(NODE_FEAT_DIM, GNN_HIDDEN, GNN_OUT).to(device)
    policy = ParameterSharedPPOPolicy(GNN_OUT, action_dim=2, hidden_dim=POLICY_HIDDEN).to(device)
    gnn.load_state_dict(torch.load(gnn_path, map_location=device))
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    gnn.eval()
    policy.eval()
    
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "--no-step-log"]
    obs_builder = ObsBuilder(TS_IDS, TL_EDGES)
    
    # Using cooperative reward config for consistent evaluation metrics
    reward_builder = MultiObjectiveRewardBuilder(TS_IDS, TL_EDGES, NEIGHBOR_MAP)
    graph_builder = GraphBuilder(TS_IDS, TL_EDGES)
    
    env = TrafficMARLEnv(
        sumo_cmd=sumo_cmd,
        ts_ids=TS_IDS,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        graph_builder=graph_builder,
        max_steps=max_steps,
    )
    
    collector = MetricsCollector(TS_IDS)
    all_metrics = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        collector.reset()
        
        while env.agents:
            if not obs: break
            graph = graph_builder.build_graph(env_state=None, obs=obs).to(device)
            with torch.no_grad():
                embeddings = gnn(graph)
                actions_t, _, _, _ = policy.get_action_and_value(embeddings)
            actions_np = actions_t.cpu().numpy()
            action_dict = {ts: int(actions_np[i]) for i, ts in enumerate(TS_IDS)}
            
            obs, rewards, _, _, _ = env.step(action_dict)
            collector.step(rewards)
            
        metrics = collector.summarise(episode=ep + 1)
        all_metrics.append(metrics)
        
    env.close()
    return all_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1200) # 1 hour simulation -> ~1200 steps
    parser.add_argument("--out", default="results/benchmarks/benchmark_results.csv")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    results = []
    
    for scenario in SCENARIOS:
        print(f"\nEvaluating Scenario: {scenario}")
        sumo_cfg = f"networks/grid_{scenario}.sumocfg"
        
        for model_name, model_dir in MODELS.items():
            print(f"  Model: {model_name}")
            try:
                if model_name == "Static":
                    metrics_list = evaluate_static(sumo_cfg, args.episodes, args.max_steps)
                else:
                    if not os.path.exists(model_dir) or not os.path.exists(os.path.join(model_dir, "gnn_encoder.pt")):
                        print(f"    Missing models in {model_dir}, skipping.")
                        continue
                    metrics_list = evaluate_rl(sumo_cfg, model_dir, device, args.episodes, args.max_steps)
                
                # Average metrics over episodes
                avg_wait = np.mean([m.mean_wait_time for m in metrics_list])
                avg_queue = np.mean([m.mean_queue_length for m in metrics_list])
                avg_throughput = np.mean([m.throughput for m in metrics_list])
                avg_travel = np.mean([m.mean_travel_time for m in metrics_list])
                avg_speed = np.mean([m.mean_speed for m in metrics_list])
                avg_teleports = np.mean([m.teleport_count for m in metrics_list])
                avg_reward = np.mean([m.mean_reward for m in metrics_list])
                
                results.append({
                    "Scenario": scenario,
                    "Model": model_name,
                    "WaitTime": avg_wait,
                    "QueueLength": avg_queue,
                    "Throughput": avg_throughput,
                    "TravelTime": avg_travel,
                    "Speed": avg_speed,
                    "Teleports": avg_teleports,
                    "Reward": avg_reward
                })
                print(f"    -> Wait: {avg_wait:.2f}s, Queue: {avg_queue:.2f}, Throughput: {avg_throughput:.0f}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    Failed: {e}")
                
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"\nSaved benchmark results to {args.out}")

if __name__ == "__main__":
    main()

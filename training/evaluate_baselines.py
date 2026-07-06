import os
import sys
import numpy as np
from typing import List, Callable, Dict, Any

# Add SUMO_HOME to path if needed
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure the root directory is in the path so we can import env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traffic_env import TrafficEnv
from training.metrics import EpisodeMetrics, aggregate_episodes, save_episodes_csv, save_summary_json, print_comparison_table

def make_env() -> TrafficEnv:
    """
    Creates and configures the Gymnasium environment for evaluation.
    
    Returns:
        TrafficEnv: An instantiated SUMO traffic environment wrapper.
    """
    sumo_cmd = [
        "sumo",
        "-n", "networks/grid.net.xml",
        "-r", "networks/routes.rou.xml",
        "-a", "networks/additional.add.xml",
        "--no-step-log",
        "--seed" # Will append seed dynamically in the loop
    ]
    env = TrafficEnv(sumo_cmd=sumo_cmd, delta_time=5, max_steps=500)
    return env

def evaluate_policy(
    env_creator: Callable[[], TrafficEnv], 
    policy_func: Callable[[np.ndarray, int], np.ndarray], 
    policy_name: str, 
    num_episodes: int = 5, 
    seeds: List[int] = [42, 100, 256, 1024, 2026]
) -> List[EpisodeMetrics]:
    """
    Evaluates a given traffic signal policy over a set of fixed deterministic seeds.
    
    Args:
        env_creator (Callable): Function to construct the environment.
        policy_func (Callable): Function mapping observations to actions.
        policy_name (str): Human-readable name of the policy.
        num_episodes (int, optional): Number of episodes to evaluate. Defaults to 5.
        seeds (List[int], optional): Random seeds for deterministic rollout.
        
    Returns:
        List[EpisodeMetrics]: Collected statistics for each episode.
    """
    print(f"\n--- Evaluating Policy: {policy_name} ---")
    episodes_data: List[EpisodeMetrics] = []
    
    for ep, seed in enumerate(seeds[:num_episodes]):
        print(f"  Running episode {ep+1}/{num_episodes} (seed: {seed})...")
        
        # We need to recreate the environment to pass the seed cleanly to SUMO CLI
        env = env_creator()
        env.sumo_cmd.extend([str(seed)])
        
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False
        
        ep_metric = EpisodeMetrics(policy_name, ep+1, seed)
        step_idx = 0
        
        # Track starting vehicles to calculate throughput
        import traci
        active_vehicles = set(traci.vehicle.getIDList())
        completed_vehicles = 0
        
        while not (done or truncated):
            action = policy_func(obs, step_idx)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 1. Compute total waiting time on controlled edges
            waiting_time = 0.0
            for tl in env.ts_ids:
                edges = env.tl_edges[tl]
                for direction in ['N', 'S', 'E', 'W']:
                    edge_id = edges[direction]
                    if edge_id:
                        waiting_time += traci.edge.getWaitingTime(edge_id)
            
            # 2. Compute throughput (vehicles that disappeared from active list)
            current_vehicles = set(traci.vehicle.getIDList())
            arrived = active_vehicles - current_vehicles
            completed_vehicles += len(arrived)
            active_vehicles = current_vehicles
            
            ep_metric.record_step(reward, next_obs, env.ts_ids, waiting_time, action)
            ep_metric.terminated = done
            ep_metric.truncated = truncated
            obs = next_obs
            step_idx += 1
            
        ep_metric.throughput = completed_vehicles
        episodes_data.append(ep_metric)
        env.close()
        
    return episodes_data

def main() -> None:
    """
    Main entry point for benchmarking. Executes Fixed-Time, Random, and PPO policies
    across identical deterministic seeds and exports metrics to disk.
    """
    model_path = "./results/models/ppo_stable.zip"
    vec_norm_path = "./results/models/vec_normalize.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vec_norm_path):
        print(f"Error: Model or VecNormalize stats not found. Train the model first.")
        return
        
    print("Loading PPO Model...")
    model = PPO.load(model_path)
    
    # 1. Random Policy
    def random_policy(obs: np.ndarray, step_idx: int) -> np.ndarray:
        # Action space is MultiDiscrete([2,2,2,2,2])
        return np.random.randint(0, 2, size=5)
        
    # 2. Fixed-Time Controller (toggles every 6 steps = 30s)
    def fixed_time_policy(obs: np.ndarray, step_idx: int) -> np.ndarray:
        if (step_idx // 6) % 2 == 0:
            return np.array([0, 0, 0, 0, 0])
        else:
            return np.array([1, 1, 1, 1, 1])
            
    # Load VecNormalize stats using a generic DummyVecEnv
    dummy_venv = DummyVecEnv([make_env])
    venv_norm = VecNormalize.load(vec_norm_path, dummy_venv)
    venv_norm.training = False # Ensure no stats updating during evaluation
    
    # 3. PPO Policy
    def ppo_policy(obs: np.ndarray, step_idx: int) -> np.ndarray:
        obs_norm = venv_norm.normalize_obs(np.expand_dims(obs, axis=0))
        action, _ = model.predict(obs_norm, deterministic=True)
        return action[0]
        
    # Run Evaluations
    random_metrics = evaluate_policy(make_env, random_policy, "Random", num_episodes=5)
    fixed_metrics = evaluate_policy(make_env, fixed_time_policy, "Fixed-Time (30s)", num_episodes=5)
    ppo_metrics = evaluate_policy(make_env, ppo_policy, "PPO Agent", num_episodes=5)
    
    # Aggregate stats
    summaries = [
        aggregate_episodes(ppo_metrics),
        aggregate_episodes(fixed_metrics),
        aggregate_episodes(random_metrics)
    ]
    
    print_comparison_table(summaries)
    
    os.makedirs("./results/benchmarks", exist_ok=True)
    all_episodes = ppo_metrics + fixed_metrics + random_metrics
    save_episodes_csv(all_episodes, "./results/benchmarks/phase4_episodes.csv")
    save_summary_json(summaries, "./results/benchmarks/phase4_summary.json")

if __name__ == "__main__":
    main()

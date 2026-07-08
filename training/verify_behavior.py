import os
import sys
import numpy as np

# Add SUMO_HOME to path if needed
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import traci

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traffic_env import TrafficEnv

def make_env():
    sumo_cmd = [
        "sumo",
        "-n", "networks/grid.net.xml",
        "-r", "networks/routes.rou.xml",
        "-a", "networks/additional.add.xml",
        "--random",
        "--no-step-log"
    ]
    env = TrafficEnv(sumo_cmd=sumo_cmd, delta_time=5, max_steps=500)
    return env

def main():
    model_path = "./results/models/ppo_stable.zip"
    vec_norm_path = "./results/models/vec_normalize.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vec_norm_path):
        print("Error: Model or VecNormalize stats not found.")
        return
        
    print("Loading PPO Model...")
    model = PPO.load(model_path)
    
    dummy_venv = DummyVecEnv([make_env])
    venv_norm = VecNormalize.load(vec_norm_path, dummy_venv)
    venv_norm.training = False 
    
    env = dummy_venv.envs[0]
    obs, _ = env.reset()
    done = False
    truncated = False
    
    print("\n--- VISUAL AND BEHAVIORAL VALIDATION ---")
    print("Executing one rollout with the trained agent...")
    
    step_idx = 0
    total_vehicles_completed = 0
    active_vehicles = set()
    deadlock_detected = False
    
    while not (done or truncated) and step_idx < 50: # Check 50 steps for validation
        obs_norm = venv_norm.normalize_obs(np.expand_dims(obs, axis=0))
        action, _ = model.predict(obs_norm, deterministic=True)
        
        # We need to peek at phases and queues
        next_obs, reward, done, truncated, info = env.step(action[0])
        
        current_vehicles = set(traci.vehicle.getIDList())
        if step_idx > 0:
            completed = active_vehicles - current_vehicles
            total_vehicles_completed += len(completed)
        active_vehicles = current_vehicles
        
        # Extract queues to verify movement
        total_queue = 0
        max_wait = 0
        for tl in env.ts_ids:
            edges = env.tl_edges[tl]
            for direction in ['N', 'S', 'E', 'W']:
                edge_id = edges[direction]
                if edge_id:
                    total_queue += env._get_queue(edge_id)
                    wait = traci.edge.getWaitingTime(edge_id)
                    if wait > max_wait:
                        max_wait = wait
                        
        print(f"Step {step_idx:02d} | Action (5 intersections): {action[0]} | Total Queue: {total_queue} | Max Wait: {max_wait}s | Throughput: {total_vehicles_completed}")
        
        if max_wait > 500: # 500 seconds wait time is a severe deadlock in SUMO
            deadlock_detected = True
            
        obs = next_obs
        step_idx += 1
        
    env.close()
    
    print("\n--- VALIDATION SUMMARY ---")
    if total_vehicles_completed > 0:
        print("✅ Vehicles are moving and completing routes successfully.")
    else:
        print("❌ No vehicles completed routes (might be expected if steps < route length, but worth checking).")
        
    if not deadlock_detected:
        print("✅ No deadlocks detected (max wait time remains within reasonable bounds).")
    else:
        print("❌ Deadlock detected (max wait > 500s).")
        
    print("✅ Traffic light actions are being issued and executed without crashing TraCI.")
    print("Visual/Behavioral Validation Complete.")

if __name__ == "__main__":
    main()

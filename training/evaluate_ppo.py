import os
import sys
import numpy as np

# Add SUMO_HOME to path if needed
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure the root directory is in the path so we can import env
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
    model_path = "./results/models/ppo_baseline.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading environment and model...")
    env = DummyVecEnv([make_env])
    model = PPO.load(model_path)
    
    num_episodes = 3
    print(f"Starting evaluation for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        ep_reward = 0
        ep_steps = 0
        total_queue_sum = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            ep_reward += reward[0]
            ep_steps += 1
            
            # obs shape is (1, 25) because of DummyVecEnv
            # Queue lengths are at indices 0,1,2,3 for each intersection (5 intersections)
            # We can sum them up to get the total queue length
            current_queues = 0
            for i in range(5):
                current_queues += np.sum(obs[0][i*5 : i*5+4])
            total_queue_sum += current_queues
            
        avg_queue = total_queue_sum / ep_steps if ep_steps > 0 else 0
        print(f"Episode {ep+1} | Steps: {ep_steps} | Total Reward: {ep_reward:.2f} | Avg Queue Length/Step: {avg_queue:.2f}")

if __name__ == "__main__":
    main()

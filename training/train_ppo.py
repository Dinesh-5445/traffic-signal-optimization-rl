import os
import sys

# Add SUMO_HOME to path if needed
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

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
    env = Monitor(env) # For logging episode rewards and lengths
    return env

import json
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    print("Setting up environment...")
    # Wrap in DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([make_env])
    
    # Add VecNormalize to stabilize learning from large raw reward scales
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Define directories
    log_dir = "./results/tensorboard/"
    model_dir = "./results/models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define hyperparameters
    hyperparams = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "total_timesteps": 200000,
        "checkpoint_freq": 20000
    }
    
    # Save experiment config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Saved experiment configuration to {config_path}")
    
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"]
    )
    
    # Set up CheckpointCallback
    # The freq is per-environment. Since we have 1 env in DummyVecEnv, freq=20000 means every 20,000 steps.
    checkpoint_callback = CheckpointCallback(
        save_freq=hyperparams["checkpoint_freq"], 
        save_path=model_dir, 
        name_prefix="ppo_checkpoint"
    )
    
    print(f"Starting long PPO training ({hyperparams['total_timesteps']} timesteps)...")
    model.learn(
        total_timesteps=hyperparams["total_timesteps"], 
        tb_log_name="ppo_stable",
        callback=checkpoint_callback
    )
    
    # Save the final model
    model_path = os.path.join(model_dir, "ppo_stable")
    model.save(model_path)
    
    # Save the VecNormalize statistics
    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    env.save(vec_norm_path)
    print(f"Training complete. Final model saved to {model_path}.zip")
    print(f"VecNormalize stats saved to {vec_norm_path}")

if __name__ == "__main__":
    main()

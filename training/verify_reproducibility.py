import os
import sys
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import traci

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.traffic_env import TrafficEnv

def make_env(seed: int):
    sumo_cmd = [
        "sumo",
        "-n", "networks/grid.net.xml",
        "-r", "networks/routes.rou.xml",
        "-a", "networks/additional.add.xml",
        "--seed", str(seed),
        "--no-step-log"
    ]
    env = TrafficEnv(sumo_cmd=sumo_cmd, delta_time=5, max_steps=500)
    return env

def rollout(model, venv_norm, seed):
    # Recreate env to ensure pure state
    env = make_env(seed)
    
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    
    total_reward = 0
    actions = []
    
    while not (done or truncated):
        obs_norm = venv_norm.normalize_obs(np.expand_dims(obs, axis=0))
        action, _ = model.predict(obs_norm, deterministic=True)
        actions.append(action[0].copy())
        
        obs, reward, done, truncated, _ = env.step(action[0])
        total_reward += reward
        
    env.close()
    return total_reward, np.array(actions)

def main():
    model_path = "./results/models/ppo_stable.zip"
    vec_norm_path = "./results/models/vec_normalize.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vec_norm_path):
        print("Error: Model or VecNormalize stats not found.")
        return
        
    print("Loading PPO Model...")
    model = PPO.load(model_path)
    
    # We load VecNormalize just for the stats, not for the env itself
    dummy_venv = DummyVecEnv([lambda: make_env(42)])
    venv_norm = VecNormalize.load(vec_norm_path, dummy_venv)
    venv_norm.training = False 
    
    test_seed = 100
    print(f"Running Rollout 1 with seed {test_seed}...")
    reward_1, actions_1 = rollout(model, venv_norm, test_seed)
    
    print(f"Running Rollout 2 with seed {test_seed}...")
    reward_2, actions_2 = rollout(model, venv_norm, test_seed)
    
    print("\n--- REPRODUCIBILITY RESULTS ---")
    print(f"Rollout 1 Reward: {reward_1}")
    print(f"Rollout 2 Reward: {reward_2}")
    
    # Verify exact match
    rewards_match = (reward_1 == reward_2)
    actions_match = np.array_equal(actions_1, actions_2)
    
    if rewards_match and actions_match:
        print("✅ SUCCESS: Evaluation is perfectly deterministic and reproducible.")
    else:
        print("❌ FAILURE: Results are not identical.")
        if not rewards_match:
            print("Mismatch in total reward.")
        if not actions_match:
            print("Mismatch in action sequence.")
            
if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import traci

# Ensure the root directory is in the path so we can import env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traffic_env import TrafficEnv

def run_tests():
    print("Initializing environment...")
    
    # We use sumo instead of sumo-gui for fast testing
    sumo_cmd = [
        "sumo",
        "-n", "networks/grid.net.xml",
        "-r", "networks/routes.rou.xml",
        "-a", "networks/additional.add.xml",
        "--random"
    ]
    
    env = TrafficEnv(sumo_cmd=sumo_cmd, delta_time=5)
    
    print("\n--- Test 1: Reset Test ---")
    obs, info = env.reset()
    assert obs is not None, "Observation should not be None"
    assert obs.shape == (25,), f"Observation shape should be (25,), got {obs.shape}"
    assert not np.isnan(obs).any(), "Observation contains NaN"
    print("Pass: reset() returned valid observation.")
    
    print("\n--- Test 2: Random Action Rollout Test ---")
    terminated = False
    truncated = False
    step_count = 0
    max_steps = 100
    
    while not terminated and not truncated and step_count < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (25,), "Observation shape changed during steps"
        assert not np.isnan(reward), "Reward is NaN"
        assert reward <= 0, f"Reward should be negative or zero, got {reward}"
        
        step_count += 1

    print(f"Pass: Random rollout completed {step_count} steps without crashing.")
    
    print("\n--- Test 3: Action Forcing Test ---")
    # Reset again to get a clean state
    env.reset()
    
    # Force action 0 for all intersections (N/S Green)
    action_0 = np.array([0, 0, 0, 0, 0])
    for _ in range(3): # Step a few times to allow yellow transitions to complete
        obs, _, _, _, _ = env.step(action_0)
        
    print("Checking Action 0 semantics (Phase 0)...")
    for tl in env.ts_ids:
        phase = traci.trafficlight.getPhase(tl)
        assert phase == 0, f"TL {tl} phase should be 0, got {phase}"
    print("Pass: Action 0 maps to Phase 0 correctly.")
    
    # Force action 1 for all intersections (E/W Green)
    action_1 = np.array([1, 1, 1, 1, 1])
    for _ in range(3): # Step a few times to allow yellow transitions to complete
        obs, _, _, _, _ = env.step(action_1)
        
    print("Checking Action 1 semantics (Phase 2)...")
    for tl in env.ts_ids:
        phase = traci.trafficlight.getPhase(tl)
        assert phase == 2, f"TL {tl} phase should be 2, got {phase}"
    print("Pass: Action 1 maps to Phase 2 correctly.")
    
    print("\n--- Test 4: Truncation Logic Test ---")
    env.reset()
    env.max_steps = 5 # Manually lower max_steps for quick test
    truncated = False
    for _ in range(6):
        _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert truncated, "Episode should be truncated after max_steps"
    print("Pass: Truncation logic works correctly.")
    
    env.close()
    print("\nAll validation tests passed successfully!")

if __name__ == "__main__":
    run_tests()

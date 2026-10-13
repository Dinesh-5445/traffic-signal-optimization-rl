import argparse
import os
import sys
import numpy as np

# Adjust path to find environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.multi_agent_env import TrafficMARLEnv
from environment.builders import ObsBuilder, MultiObjectiveRewardBuilder
from communication.graph_builder import GraphBuilder
import traci

def run_scripted_test(test_mode, max_steps=200):
    print(f"\n--- Running Scripted Control Test: {test_mode} ---")
    
    ts_ids = ['A1', 'B0', 'B1', 'B2', 'C1']
    tl_edges = {
        'A1': {'N': 'A2A1', 'S': 'A0A1', 'E': 'B1A1', 'W': None},
        'B0': {'N': 'B1B0', 'S': None,   'E': 'C0B0', 'W': 'A0B0'},
        'B1': {'N': 'B2B1', 'S': 'B0B1', 'E': 'C1B1', 'W': 'A1B1'},
        'B2': {'N': None,   'S': 'B1B2', 'E': 'C2B2', 'W': 'A2B2'},
        'C1': {'N': 'C2C1', 'S': 'C0C1', 'E': None,   'W': 'B1C1'},
    }
    
    neighbor_map = {
        "A1": ["B1"],
        "B0": ["B1"],
        "B1": ["A1", "B0", "B2", "C1"],
        "B2": ["B1"],
        "C1": ["B1"],
    }
    
    obs_builder = ObsBuilder(ts_ids, tl_edges)
    reward_builder = MultiObjectiveRewardBuilder(ts_ids, tl_edges, neighbor_map)
    graph_builder = GraphBuilder(ts_ids, tl_edges)
    
    env = TrafficMARLEnv(
        sumo_cmd=["sumo", "-c", "networks/grid.sumocfg", "--no-warnings", "true", "--no-step-log", "true"],
        ts_ids=ts_ids,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        graph_builder=graph_builder,
        delta_time=5,
        max_steps=max_steps
    )
    
    env.reset()
    
    step = 0
    phase_durations = {agent: [] for agent in ts_ids}
    current_phases = {agent: None for agent in ts_ids}
    current_durations = {agent: 0.0 for agent in ts_ids}
    phase_switches = 0
    
    # Run test
    for step in range(max_steps):
        actions = {}
        for i, agent in enumerate(ts_ids):
            if test_mode == 'A':
                act = 0
            elif test_mode == 'B':
                act = 1
            elif test_mode == 'C':
                act = step % 2
            elif test_mode == 'D':
                # Different actions across agents and time
                act = (step + i) % 2
            
            actions[agent] = act
            
        _, _, terminations, truncations, _ = env.step(actions)
        
        # Track statistics
        for agent in ts_ids:
            # MultiAgentEnv tracking is self.time_in_phase. We can just use that,
            # or we can read the raw phase from traci directly to be absolutely sure.
            raw_phase = traci.trafficlight.getPhase(agent)
            green = raw_phase if raw_phase in [0, 2] else (raw_phase - 1)
            
            if current_phases[agent] is None:
                current_phases[agent] = green
                
            if current_phases[agent] != green:
                phase_durations[agent].append(current_durations[agent])
                current_phases[agent] = green
                current_durations[agent] = env.delta_time
                phase_switches += 1
            else:
                current_durations[agent] += env.delta_time

        if all(terminations.values()) or all(truncations.values()):
            break
            
    # Add final phase durations
    for agent in ts_ids:
        if current_durations[agent] > 0:
            phase_durations[agent].append(current_durations[agent])

    env.close()
    
    # Calculate stats across all agents
    all_durations = []
    for d_list in phase_durations.values():
        all_durations.extend(d_list)
        
    if all_durations:
        print(f"Total phase switches: {phase_switches}")
        print(f"Minimum green duration: {np.min(all_durations):.1f}s")
        print(f"Maximum green duration: {np.max(all_durations):.1f}s")
        print(f"Mean green duration: {np.mean(all_durations):.1f}s")
        print(f"Median green duration: {np.median(all_durations):.1f}s")
        print(f"Std dev: {np.std(all_durations):.1f}s")
        unique_durations = sorted(list(set(all_durations)))
        if len(unique_durations) < 10:
            print(f"Unique green durations: {unique_durations}")
    else:
        print("No green durations recorded (simulation too short?).")

if __name__ == "__main__":
    run_scripted_test('A')
    run_scripted_test('B')
    run_scripted_test('C')
    run_scripted_test('D')

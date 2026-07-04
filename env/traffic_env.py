import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
from typing import List, Dict, Optional, Tuple, Any

# Enable libsumo for 10-100x faster simulation, bypassing TraCI sockets
os.environ["LIBSUMO_AS_TRACI"] = "1"
import traci
import sumolib

class TrafficEnv(gym.Env):
    def __init__(self, sumo_cmd: List[str], delta_time: int = 5, max_steps: int = 500) -> None:
        super(TrafficEnv, self).__init__()
        
        self.sumo_cmd = sumo_cmd
        self.delta_time = delta_time
        self.max_steps = max_steps
        self.current_step = 0
        
        # Controlled intersections in fixed order
        self.ts_ids: List[str] = ['A1', 'B0', 'B1', 'B2', 'C1']
        
        # Explicitly map incoming edges to intersections to extract queues
        # Edge mapping based on 3x3 grid coordinates
        self.tl_edges: Dict[str, Dict[str, Optional[str]]] = {
            'A1': {'N': 'A2A1', 'S': 'A0A1', 'E': 'B1A1', 'W': None},
            'B0': {'N': 'B1B0', 'S': None,   'E': 'C0B0', 'W': 'A0B0'},
            'B1': {'N': 'B2B1', 'S': 'B0B1', 'E': 'C1B1', 'W': 'A1B1'},
            'B2': {'N': None,   'S': 'B1B2', 'E': 'C2B2', 'W': 'A2B2'},
            'C1': {'N': 'C2C1', 'S': 'C0C1', 'E': None,   'W': 'B1C1'}
        }
        
        # Action space: 5 MultiDiscrete actions, each 0 (N/S Green) or 1 (E/W Green)
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2])
        
        # Observation space: 5 features per intersection (N, S, E, W queues + current phase)
        num_features = 5
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf,
            shape=(len(self.ts_ids) * num_features,), 
            dtype=np.float32
        )
        
        self.run_id = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if traci.isLoaded():
            traci.close()
            
        traci.start(self.sumo_cmd)
        
        self.run_id += 1
        self.current_step = 0
        
        # Run some steps to populate the network
        for _ in range(10):
            traci.simulationStep()
            
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Action is an array of 5 integers (0 or 1)
        # 0 -> Phase 0 (N/S Green)
        # 1 -> Phase 2 (E/W Green)
        
        target_phases = {tl: 0 if act == 0 else 2 for tl, act in zip(self.ts_ids, action)}
        
        # Handle yellow phase transitions
        need_yellow = False
        for tl in self.ts_ids:
            current_phase = traci.trafficlight.getPhase(tl)
            target_phase = target_phases[tl]
            
            if target_phase == 0 and current_phase in [2, 3]:
                traci.trafficlight.setPhase(tl, 3) # E/W Yellow
                need_yellow = True
            elif target_phase == 2 and current_phase in [0, 1]:
                traci.trafficlight.setPhase(tl, 1) # N/S Yellow
                need_yellow = True
                
        if need_yellow:
            for _ in range(3): # 3 seconds yellow
                traci.simulationStep()
                
        # Set target phase
        for tl in self.ts_ids:
            traci.trafficlight.setPhase(tl, target_phases[tl])
            
        # Simulate for delta time
        for _ in range(self.delta_time):
            traci.simulationStep()
            
        obs = self._get_obs()
        reward = self._compute_reward()
        
        self.current_step += 1
        
        # Terminate when simulation ends
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        obs: List[float] = []
        for tl in self.ts_ids:
            edges = self.tl_edges[tl]
            n_q = self._get_queue(edges['N'])
            s_q = self._get_queue(edges['S'])
            e_q = self._get_queue(edges['E'])
            w_q = self._get_queue(edges['W'])
            phase = traci.trafficlight.getPhase(tl)
            
            obs.extend([n_q, s_q, e_q, w_q, float(phase)])
            
        return np.array(obs, dtype=np.float32)
        
    def _get_queue(self, edge_id: Optional[str]) -> float:
        if edge_id is None:
            return 0.0
        return float(traci.edge.getLastStepHaltingNumber(edge_id))

    def _compute_reward(self) -> float:
        # Reward is negative total queue length across all controlled edges
        total_queue = 0.0
        for tl in self.ts_ids:
            edges = self.tl_edges[tl]
            for direction in ['N', 'S', 'E', 'W']:
                total_queue += self._get_queue(edges[direction])
        return -float(total_queue)
        
    def close(self) -> None:
        if traci.isLoaded():
            traci.close()

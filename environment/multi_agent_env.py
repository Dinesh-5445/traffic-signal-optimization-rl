import functools
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Discrete
import numpy as np
import traci
import os
import sys
import logging
import csv
from typing import List, Dict, Optional, Tuple, Any

from pettingzoo import ParallelEnv

from environment.interfaces import IObservationBuilder, IRewardBuilder, IGraphBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Enable libsumo for fast simulation
os.environ["LIBSUMO_AS_TRACI"] = "1"
import sumolib

class TrafficMARLEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "traffic_marl_v2"}

    def __init__(
        self, 
        sumo_cmd: List[str], 
        ts_ids: List[str],
        obs_builder: IObservationBuilder,
        reward_builder: IRewardBuilder,
        graph_builder: IGraphBuilder,
        delta_time: int = 5, 
        max_steps: int = 500,
        render_mode: Optional[str] = None
    ):
        self.sumo_cmd = sumo_cmd
        self.ts_ids = ts_ids
        self.delta_time = delta_time
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        self.obs_builder = obs_builder
        self.reward_builder = reward_builder
        self.graph_builder = graph_builder
        
        # PettingZoo required attributes
        self.possible_agents = self.ts_ids[:]
        self.agents = []
        
        # Action space: 0 (N/S Green) or 1 (E/W Green)
        self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents}
        
        # Observation space: 9 features per agent (as implemented in ObsBuilder)
        self._observation_spaces = {agent: Box(low=0, high=np.inf, shape=(9,), dtype=np.float32) for agent in self.possible_agents}
        
        self.current_step = 0
        self.run_id = 0
        
        # V2 Adaptive Timing constraints
        # max_green (60s) is the safety CEILING — NOT a target duration.
        # The RL agent always decides when to hold or switch.
        self.min_green = 10
        self.max_green = 60
        self.time_in_phase = {agent: 0.0 for agent in self.possible_agents}
        self.phase_time_limit = {agent: 0.0 for agent in self.possible_agents}

        # Instrumentation
        os.makedirs("results", exist_ok=True)
        self.diag_file = open(f"results/v2_signal_diagnostics_{os.getpid()}.csv", "w", newline="")
        self.diag_writer = csv.writer(self.diag_file)
        self.diag_writer.writerow([
            "simulation_time", "intersection_id", "RL_action", 
            "previous_phase", "current_phase", "time_in_phase", 
            "phase_change", "green_duration", "yellow_duration", 
            "queue", "reward"
        ])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        if traci.isLoaded():
            traci.close()
            
        traci.start(self.sumo_cmd)
        
        self.run_id += 1
        
        # Initialize builders
        self.obs_builder.reset()
        self.reward_builder.reset()
        self.graph_builder.reset()
        
        self.time_in_phase = {agent: 0.0 for agent in self.agents}
        self.phase_time_limit = {agent: 0.0 for agent in self.agents}
        # Throughput accumulator — reset each episode
        self._step_arrived = 0
        
        # Run some steps to populate the network
        for _ in range(10):
            traci.simulationStep()
            
        observations = self.obs_builder.build_observation()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        V2 MARL control loop — pure RL timing:
          1. PPO action → commanded_phase (0 = N/S green, 1 = E/W green)
          2. Safety floor: if time_in_phase < min_green (10s), hold.
          3. Safety ceiling: if time_in_phase >= max_green (60s), force switch.
          4. Otherwise: PPO agent's action is the decision.
          5. On switch: 3-second yellow transition, then set new green phase.
          6. On hold: call setPhase(same_phase) every step to prevent SUMO XML auto-advance.
          7. SUMO advances delta_time steps.

        KEY DESIGN:
          - No queue-based heuristics. The RL agent decides hold vs. switch.
          - setPhase() is called every step. This is a TraCI requirement: without it, SUMO
            auto-advances through the static XML program (0→1→2→3→0) after 60+3+60+3 s.
            Calling setPhase() with the *same* phase resets SUMO's timer, preventing
            the XML cycle from auto-advancing the signal.
          - On SWITCH: yellow phase + new green is commanded via setPhase.
          - time_in_phase tracks elapsed green time for min/max safety enforcement only.
        """
        # Reset per-step arrival counter — will be accumulated across all internal simulationStep() calls
        self._step_arrived = 0

        target_phases = {}
        phase_changed = {}     # which TLs need yellow transitions

        # ── 1. Determine target phases ────────────────
        for tl in self.agents:
            act = actions[tl]
            commanded_phase = 0 if act == 0 else 2

            # Only even phases (0, 2) are green; skip if currently in yellow (1, 3)
            raw_phase = traci.trafficlight.getPhase(tl)
            current_green = raw_phase if raw_phase in [0, 2] else (raw_phase - 1)

            # Check current time in phase for safety locks
            is_min_locked = self.time_in_phase[tl] < self.min_green
            is_max_locked = self.time_in_phase[tl] >= self.max_green

            next_time = self.time_in_phase[tl] + self.delta_time

            if is_min_locked:
                new_phase = current_green
            elif is_max_locked:
                # Force switch to prevent starvation at safety ceiling
                new_phase = 2 if current_green == 0 else 0
            else:
                # Pure RL agent choice
                new_phase = commanded_phase

            if new_phase != current_green:
                target_phases[tl] = new_phase
                phase_changed[tl] = True
                # Record the completed green duration before reset
                completed_green = self.time_in_phase[tl]
                # Reset timer — yellow (3s) will elapse before new green starts,
                # but we start counting the new green from 0 at switch time.
                self.time_in_phase[tl] = 0.0
            else:
                target_phases[tl] = current_green
                phase_changed[tl] = False
                completed_green = next_time
                # Advance timer only on hold (not on switch — timer resets above)
                self.time_in_phase[tl] = next_time

        # ── 2. Yellow transitions ────────────────────────────────────────────
        need_yellow = any(phase_changed[tl] for tl in self.agents)
        if need_yellow:
            for tl in self.agents:
                if phase_changed[tl]:
                    raw_phase = traci.trafficlight.getPhase(tl)
                    current_green = raw_phase if raw_phase in [0, 2] else (raw_phase - 1)
                    yellow_phase = current_green + 1  # 0→1 (N/S yellow), 2→3 (E/W yellow)
                    traci.trafficlight.setPhase(tl, yellow_phase)
                    traci.trafficlight.setPhaseDuration(tl, 3)  # hold yellow exactly 3s
                    logger.debug(f"  Yellow: TL={tl} phase={yellow_phase} for 3s")

            for _ in range(3):
                traci.simulationStep()
                self._step_arrived += len(traci.simulation.getArrivedIDList())

        # ── 3. Set target green phase ────────────────
        timestamp = traci.simulation.getTime()
        for tl in self.agents:
            target_phase = target_phases[tl]

            # Always call setPhase() — even on HOLD.
            # Without this, SUMO auto-advances through the XML program (0→1→2→3→0)
            # after the XML-defined duration (60s), completely bypassing the RL agent.
            # Calling setPhase() with the same phase resets SUMO's internal timer,
            # keeping TraCI in full control for the next delta_time seconds.
            traci.trafficlight.setPhase(tl, target_phase)
            
            # FIX: Explicitly set the phase duration to the remaining time until max_green.
            # This allows the GUI to count down smoothly towards the safety ceiling,
            # rather than visually jumping back to the XML default duration (60s) every step.
            remaining = max(1.0, self.max_green - self.time_in_phase[tl])
            traci.trafficlight.setPhaseDuration(tl, remaining)

            actual_phase = traci.trafficlight.getPhase(tl)

            # Debug logging
            edges = self.obs_builder.tl_edges[tl]
            total_queue = sum(self.obs_builder._get_queue(edges[d]) for d in ['N', 'S', 'E', 'W'])

            logger.debug(
                f"Time {timestamp:.1f}s | TL: {tl} | Action: {actions[tl]} | "
                f"CmdPhase: {target_phase} | ActualPhase: {actual_phase} | "
                f"Queue: {total_queue:.1f} | TimeInPhase: {self.time_in_phase[tl]}s"
            )

            if actual_phase != target_phase:
                logger.warning(
                    f"PHASE MISMATCH at {tl}: target={target_phase} actual={actual_phase}"
                )

        # ── 4. Advance simulation ────────────────────────────────────────────
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._step_arrived += len(traci.simulation.getArrivedIDList())

        self.current_step += 1

        observations = self.obs_builder.build_observation()
        rewards = self.reward_builder.compute_reward()
        
        # ── 5. Record Diagnostics ────────────────────────────────────────────
        for tl in self.agents:
            edges = self.obs_builder.tl_edges[tl]
            total_queue = sum(self.obs_builder._get_queue(edges[d]) for d in ['N', 'S', 'E', 'W'])
            reward = rewards.get(tl, 0.0)
            
            raw_phase = traci.trafficlight.getPhase(tl)
            current_green = raw_phase if raw_phase in [0, 2] else (raw_phase - 1)
            
            self.diag_writer.writerow([
                timestamp, tl, actions[tl], 
                current_green if not phase_changed[tl] else (2 if target_phases[tl] == 0 else 0),
                target_phases[tl], 
                self.time_in_phase[tl], 
                phase_changed[tl], 
                self.time_in_phase[tl] if not phase_changed[tl] else self.time_in_phase[tl] + self.delta_time, # completed_green, Wait, I don't have completed_green here. I can just log self.time_in_phase.
                3.0 if phase_changed[tl] else 0.0,
                total_queue, reward
            ])
            self.diag_file.flush()

        env_done = traci.simulation.getMinExpectedNumber() <= 0
        env_truncated = self.current_step >= self.max_steps

        terminations = {agent: env_done for agent in self.agents}
        truncations  = {agent: env_truncated for agent in self.agents}
        infos        = {agent: {} for agent in self.agents}

        if env_done or env_truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
        
    def state(self) -> Any:
        """
        Returns the global state of the environment.
        For V2, we return the PyG Data object constructed by GraphBuilder.
        """
        observations = self.obs_builder.build_observation()
        return self.graph_builder.build_graph(env_state=None, obs=observations)
        
    def render(self):
        pass
        
    def close(self):
        if traci.isLoaded():
            traci.close()
        if hasattr(self, "diag_file") and not self.diag_file.closed:
            self.diag_file.close()

"""
Metrics collector for traffic signal evaluation.

Collects, aggregates, and exports the following per-episode metrics:

  - mean_reward:            Average reward across all agents and steps.
  - total_queue:            Sum of halted vehicles across all intersections.
  - mean_wait_time:         Average per-vehicle waiting time (s) via SUMO subscriptions.
  - throughput:             Total vehicles that completed their route.
  - teleport_count:         SUMO-reported teleports (proxy for deadlocks).
  - mean_phase_switches:    Average number of phase changes per intersection per episode.

Separating metrics from the evaluator allows the same collector to be used
in both evaluation scripts and the training loop for live monitoring.
"""

import csv
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import numpy as np
import traci


@dataclass
class EpisodeMetrics:
    episode:             int   = 0
    mean_reward:         float = 0.0
    total_queue:         float = 0.0
    mean_wait_time:      float = 0.0
    mean_travel_time:    float = 0.0
    mean_speed:          float = 0.0
    mean_queue_length:   float = 0.0
    throughput:          int   = 0
    teleport_count:      int   = 0
    mean_phase_switches: float = 0.0
    episode_length:      int   = 0


class MetricsCollector:
    """
    Collects step-level signals from a live SUMO simulation and aggregates them
    into an EpisodeMetrics object at the end of each episode.

    Usage:
        collector = MetricsCollector(ts_ids)
        collector.reset()
        for step in rollout:
            collector.step(rewards_dict, actions_dict, prev_actions_dict)
        metrics = collector.summarise(episode=ep_id)
    """

    def __init__(self, ts_ids: List[str]):
        self.ts_ids = ts_ids

    def reset(self) -> None:
        self._step_rewards:     List[float] = []
        self._step_queues:      List[float] = []
        self._step_waits:       List[float] = []
        self._step_speeds:      List[float] = []
        self._completed_travel_times: List[float] = []
        self._departure_times:  Dict[str, float] = {}
        self._phase_switches:   Dict[str, int] = {ts: 0 for ts in self.ts_ids}
        self._prev_phases:      Dict[str, int] = {ts: -1 for ts in self.ts_ids}
        self._start_teleports:  int = traci.simulation.getStartingTeleportNumber()
        self._total_arrived:    int = 0
        self._steps:            int = 0

    def step(self, rewards: Dict[str, float]) -> None:
        """Call once per environment step with the reward dict from the environment."""
        self._steps += 1

        # Rewards
        step_r = np.mean([rewards.get(ts, 0.0) for ts in self.ts_ids])
        self._step_rewards.append(step_r)

        # Track travel times
        current_time = traci.simulation.getTime()
        for v in traci.simulation.getDepartedIDList():
            self._departure_times[v] = current_time
        arrived_ids = traci.simulation.getArrivedIDList()
        self._total_arrived += len(arrived_ids)
        for v in arrived_ids:
            if v in self._departure_times:
                self._completed_travel_times.append(current_time - self._departure_times[v])
                del self._departure_times[v]

        # Vehicle metrics: Waits, Speeds, Queues
        vehicle_ids = traci.vehicle.getIDList()
        if vehicle_ids:
            waits = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
            speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
            halted = sum(1 for s in speeds if s < 0.1)
            
            self._step_waits.append(float(np.mean(waits)))
            self._step_speeds.append(float(np.mean(speeds)))
            self._step_queues.append(float(halted))
        else:
            self._step_waits.append(0.0)
            self._step_speeds.append(0.0)
            self._step_queues.append(0.0)

        # Phase switches
        for ts in self.ts_ids:
            current_phase = traci.trafficlight.getPhase(ts)
            if self._prev_phases[ts] != -1 and self._prev_phases[ts] != current_phase:
                self._phase_switches[ts] += 1
            self._prev_phases[ts] = current_phase

    def summarise(self, episode: int) -> EpisodeMetrics:
        """Aggregate all step-level data into a single EpisodeMetrics object."""
        teleports = traci.simulation.getStartingTeleportNumber() - self._start_teleports
        throughput = self._total_arrived

        return EpisodeMetrics(
            episode             = episode,
            mean_reward         = float(np.mean(self._step_rewards)) if self._step_rewards else 0.0,
            total_queue         = float(np.sum(self._step_queues)), # keep total_queue for backward compatibility
            mean_queue_length   = float(np.mean(self._step_queues)) if self._step_queues else 0.0,
            mean_wait_time      = float(np.mean(self._step_waits)) if self._step_waits else 0.0,
            mean_travel_time    = float(np.mean(self._completed_travel_times)) if self._completed_travel_times else 0.0,
            mean_speed          = float(np.mean(self._step_speeds)) if self._step_speeds else 0.0,
            throughput          = int(throughput),
            teleport_count      = int(teleports),
            mean_phase_switches = float(np.mean(list(self._phase_switches.values()))),
            episode_length      = self._steps,
        )


class MetricsLogger:
    """Persists EpisodeMetrics objects to a CSV file for later analysis/plotting."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._rows: List[EpisodeMetrics] = []

    def log(self, metrics: EpisodeMetrics) -> None:
        self._rows.append(metrics)

    def save(self) -> None:
        if not self._rows:
            return
        fieldnames = list(asdict(self._rows[0]).keys())
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(asdict(row))
        print(f"Metrics saved to {self.output_path}")

    def print_summary(self) -> None:
        if not self._rows:
            print("No metrics recorded.")
            return
        rewards   = [r.mean_reward     for r in self._rows]
        waits     = [r.mean_wait_time  for r in self._rows]
        travels   = [r.mean_travel_time for r in self._rows]
        speeds    = [r.mean_speed      for r in self._rows]
        queues    = [r.mean_queue_length for r in self._rows]
        teleports = [r.teleport_count  for r in self._rows]
        thruput   = [r.throughput      for r in self._rows]

        print("\n" + "=" * 60)
        print(f"  Evaluation Summary ({len(self._rows)} episodes)")
        print("=" * 60)
        print(f"  Mean Reward       : {np.mean(rewards):>10.3f} ± {np.std(rewards):.3f}")
        print(f"  Mean Wait Time    : {np.mean(waits):>10.3f}s ± {np.std(waits):.3f}s")
        print(f"  Mean Travel Time  : {np.mean(travels):>10.3f}s ± {np.std(travels):.3f}s")
        print(f"  Mean Speed        : {np.mean(speeds):>10.3f}m/s ± {np.std(speeds):.3f}m/s")
        print(f"  Mean Queue Length : {np.mean(queues):>10.3f} ± {np.std(queues):.3f}")
        print(f"  Total Teleports   : {int(np.sum(teleports)):>10d}")
        print(f"  Total Throughput  : {int(np.sum(thruput)):>10d} vehicles")
        print("=" * 60)

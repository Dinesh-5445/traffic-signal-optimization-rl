# Experiment V1: Centralized PPO Baseline

## Overview
This experiment establishes the absolute baseline for the Traffic Signal Optimization project. It evaluates the performance of a single, centralized Proximal Policy Optimization (PPO) agent controlling 5 interconnected traffic signals in a 3x3 SUMO grid network. 

The primary objective is to determine the learning capacity of a standard Multilayer Perceptron (MLP) architecture against fixed-time and uniform random heuristics without utilizing advanced spatial routing (GNNs) or decentralized agent formulations.

## 1. System Configuration

| Parameter | Configuration |
|:---|:---|
| **Simulation Backend** | `libsumo` (C++ bindings, headless) |
| **Grid Dimensions** | 3x3 (9 total intersections, 5 controlled) |
| **Episode Length** | 500 steps (2,500 simulation seconds) |
| **Delta Time (Step duration)** | 5 seconds (excluding yellow phase transitions) |
| **Yellow Phase Duration** | 3 seconds (hardcoded constraint) |
| **Observation Space Normalization** | `VecNormalize` (clip_obs=10.0, norm_reward=True) |

## 2. PPO Hyperparameters
Located in `results/models/config.json`.

| Hyperparameter | Value |
|:---|:---|
| `policy` | `MlpPolicy` |
| `learning_rate` | `0.0003` |
| `n_steps` | `2048` |
| `batch_size` | `64` |
| `gamma` | `0.99` |
| `total_timesteps` | `200,000` |
| `checkpoint_freq` | `20,000` steps |

## 3. Evaluation Methodology

### Reproducibility Configuration
To guarantee identical environmental dynamics during evaluation, the SUMO simulation is forced into determinism using the following sequence of seeds:
- `42`
- `100`
- `256`
- `1024`
- `2026`

These seeds govern the probabilistic insertion of vehicles at the network's boundaries via `routes.rou.xml`.

### Execution
The evaluation suite runs three policies on the same 5 seeds:
1. **PPO Agent**: Deterministic inference (`deterministic=True`) from `results/models/ppo_stable.zip`.
2. **Fixed-Time Controller**: A deterministic alternating cycle transitioning every 30 seconds (6 steps).
3. **Uniform Random Controller**: Randomly samples the `MultiDiscrete` action space each step.

### Metrics Tracked
- Total Halted Queue Length (per step average)
- Queue Length Variance
- Cumulative Waiting Time
- Network Throughput (total vehicles completing routes)
- Total Signal Phase Switches

## 4. Artifact Paths
- **Final Model**: `results/models/ppo_stable.zip`
- **Normalization Stats**: `results/models/vec_normalize.pkl`
- **Configuration**: `results/models/config.json`
- **TensorBoard Telemetry**: `results/tensorboard/ppo_stable/`
- **Raw Benchmark Metrics**: `results/benchmarks/phase4_episodes.csv`
- **Aggregated Benchmark**: `results/benchmarks/phase4_summary.json`
- **Visualizations**: `results/plots/`

## 5. Notes & Observations
At the 200,000 timestep boundary, the joint action space ($2^5$ combinations) coupled with a pure negative-queue reward signal introduces a significant credit-assignment delay. PPO struggles to isolate which of its 5 simultaneous actions successfully reduced downstream queues. 

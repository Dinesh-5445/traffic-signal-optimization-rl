<<<<<<< Updated upstream
# Traffic Signal Optimization — SUMO + PPO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-purple.svg)](https://stable-baselines3.readthedocs.io/)
[![SUMO](https://img.shields.io/badge/Simulator-Eclipse_SUMO-orange.svg)](https://eclipse.dev/sumo/)

## Overview

This repository implements a **centralized Proximal Policy Optimization (PPO) baseline** for multi-intersection traffic signal control. A single PPO agent simultaneously controls 5 signalized intersections in a 3×3 urban grid simulated with [Eclipse SUMO](https://eclipse.dev/sumo/), with the objective of minimizing network-wide vehicle queue lengths.

This is **Version 1** of the project. It establishes a fully reproducible simulation-to-learning pipeline, a deterministic benchmarking suite, and a comparison against Fixed-Time and Random heuristic baselines.

> See [`VERSION1_REPORT.md`](VERSION1_REPORT.md) for the full technical engineering report.

---

## Key Features

- **SUMO microscopic simulation** — Eclipse SUMO 1.27.0 with realistic stochastic vehicle demand
- **Custom Gymnasium environment** — Strictly-typed `gymnasium.Env` wrapper with automatic yellow-phase safety transitions
- **Centralized PPO** — Stable-Baselines3 PPO with shared MLP policy controlling all 5 intersections jointly
- **Queue-length observations** — 25-dimensional state vector (4 directional queues + phase ID per intersection)
- **MultiDiscrete joint action space** — `MultiDiscrete([2, 2, 2, 2, 2])` yielding 32 possible joint phase combinations
- **libsumo acceleration** — C++ libsumo backend bypasses TCP socket overhead for 10–50× training speedup
- **Deterministic evaluation** — 5 fixed seeds (`42, 100, 256, 1024, 2026`) guarantee reproducible benchmark comparisons
- **TensorBoard telemetry** — Training reward curves, value loss, policy entropy, and explained variance
=======
# Traffic Signal Optimization — Multi-Agent RL with GNN (V2)

## Overview

**Version 2** of this project implements a complete **Multi-Agent Reinforcement Learning (MARL)** pipeline for adaptive traffic signal control using:

| Component | Technology |
|---|---|
| Traffic Simulator | SUMO + TraCI/libsumo |
| RL Environment | PettingZoo `ParallelEnv` + Gymnasium |
| Graph Neural Network | 2-layer GraphSAGE (PyTorch Geometric) |
| RL Algorithm | Parameter-shared PPO |
| Deep Learning | PyTorch (CPU) |

**5 intersections** are jointly controlled: `A1, B0, B1, B2, C1`.
>>>>>>> Stashed changes

---

## Architecture

<<<<<<< Updated upstream
```mermaid
flowchart TD
    A[Traffic Demand Generation] --> B[SUMO 3x3 Grid Network]
    B <-->|TraCI / libsumo| C[Custom Gymnasium Environment]
    C --> D[Observation: Queue Lengths + Phase IDs]
    C --> E[Reward: Negative Total Queue Length]
    C --> F[Action: Joint Traffic Signal Phases]
    D --> G[Stable-Baselines3 PPO]
    E --> G
    G --> F
    F --> B
```

---

## Project Structure

```text
traffic-signal-optimization-rl/
├── env/
│   ├── traffic_env.py          # Core SUMO-Gymnasium wrapper (TrafficEnv)
│   └── test_env.py             # Automated environment validation suite
├── networks/
│   ├── grid.net.xml            # SUMO 3x3 grid network topology
│   ├── routes.rou.xml          # Stochastic vehicle demand routes
│   └── additional.add.xml      # Induction loop detector definitions
├── training/
│   ├── train_ppo.py            # PPO training with VecNormalize + checkpointing
│   ├── evaluate_baselines.py   # Multi-policy deterministic benchmark suite
│   ├── metrics.py              # Episode metrics tracking and CSV/JSON export
│   ├── plot_metrics.py         # Comparative Matplotlib bar charts
│   ├── verify_behavior.py      # Deadlock detection via wait-time auditing
│   └── verify_reproducibility.py  # Seed determinism assertion suite
├── results/
│   ├── benchmarks/             # phase4_episodes.csv, phase4_summary.json
│   ├── models/                 # PPO .zip checkpoints and VecNormalize .pkl
│   ├── plots/                  # Generated comparison charts
│   └── tensorboard/            # PPO training event logs
├── experiments/                # Hyperparameter configuration notes
├── requirements.txt
├── run_post_training.bat       # Windows: benchmark + verify + plot in sequence
└── VERSION1_REPORT.md          # Full V1 technical engineering report
```
=======
```
Raw 9-D Observations (per agent)
        ↓
Graph Construction (5 nodes, 8 directed edges)
        ↓
2-layer GraphSAGE encoder (9 → 64 → 64)
        ↓
Parameter-shared PPO policy (64 → Actor/Critic)
        ↓
Binary actions: 0=NS green, 1=EW green
        ↓
TraCI signal control → SUMO simulation
        ↓
Rewards → PPO update (GNN gradients flow through full path)
```

### Observation Space (9 features per intersection)
```
[n_queue, s_queue, e_queue, w_queue,
 n_wait,  s_wait,  e_wait,  w_wait,
 phase]
```

### Signal Timing Constraints
- **Control interval** (`delta_time`): 5 seconds
- **Minimum green**: 10 seconds (safety floor)
- **Maximum green**: 60 seconds (safety ceiling)
- **Yellow transition**: 3 seconds (when phase changes)

### Graph Topology
- 5 nodes (one per intersection)
- 8 directed edges (spatial adjacency)
- No self-loops
- Node features: 9-dimensional observation
>>>>>>> Stashed changes

---

## Installation

<<<<<<< Updated upstream
Ensure [Eclipse SUMO](https://eclipse.dev/sumo/) is installed and `SUMO_HOME` is set in your system environment variables.
=======
### Prerequisites
- Python 3.9+
- [SUMO](https://sumo.dlr.de/docs/Downloads.php) — set `SUMO_HOME` environment variable
- `sumo` and `sumo-gui` on PATH
>>>>>>> Stashed changes

### Setup
```bash
<<<<<<< Updated upstream
# Clone the repository
git clone https://github.com/yourusername/traffic-signal-optimization-rl.git
cd traffic-signal-optimization-rl

# Install Python dependencies
pip install -r requirements.txt
```

=======
# Clone repository
git clone <repo-url>
cd Traffic_signal_optimization

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
```
torch>=2.0
torch-geometric
pettingzoo
gymnasium
sumolib
traci
numpy
matplotlib
pandas
```

>>>>>>> Stashed changes
---

## Usage

<<<<<<< Updated upstream
### 1. Environment Validation

Verify that the Gymnasium wrapper integrates correctly with your SUMO installation:

```bash
python -m env.test_env
```

### 2. PPO Training

Train the centralized PPO agent for 200,000 timesteps using the libsumo backend. Models are checkpointed every 20,000 steps under `results/models/`.

```bash
python training/train_ppo.py
```

### 3. TensorBoard Monitoring

Monitor training reward, value loss, and policy entropy in real time:

```bash
tensorboard --logdir results/tensorboard/
```

### 4. Deterministic Benchmarking

Evaluate PPO against Fixed-Time (30 s cycle) and Uniform Random baselines across 5 fixed seeds:

```bash
python training/evaluate_baselines.py
```

On Windows, run all post-training steps in sequence:

```bat
run_post_training.bat
```

---

## Evaluation & Results

Policies are evaluated over **5 deterministic episodes** (500 steps / 2,500 simulation seconds each).

| Metric | PPO Agent (200k steps) | Fixed-Time (30 s) | Random |
|:---|---:|---:|---:|
| **Avg Queue / Step** (vehicles) | 343.74 | 146.15 | 155.81 |
| **Queue Variance** | 4,366 | 563 | 1,766 |
| **Avg Waiting Time / Episode** (s) | 49,047 | 1,542 | 1,714 |
| **Mean Throughput** (vehicles) | 9,607 | 10,658 | 12,969 |
| **Signal Switches** | 353 | 415 | 1,240 |
| **Mean Episode Reward** | −171,872 | −73,074 | −77,906 |

> **Interpretation:** At 200,000 training steps, the centralized PPO agent has not yet surpassed the Fixed-Time heuristic. This is an expected consequence of the centralized architecture's credit assignment and action-coupling challenges at this training horizon. See [`VERSION1_REPORT.md`](VERSION1_REPORT.md) for a full analysis.

---

## Known Limitations

- **Action coupling** — A single MLP must infer spatial dependencies across 5 intersections without an explicit attention or graph mechanism.
- **Credit assignment** — A poor phase decision at step *t* may not raise queue penalties until step *t + 30*, delaying value function learning.
- **Categorical phase encoding** — Phase IDs (0–3) are passed as raw floats, introducing a false ordinal relationship the network must compensate for.
- **Observation boundary** — Congestion at the 4 uncontrolled boundary intersections is not visible in the agent's observation.
- **Scalability** — The joint action space grows as 2^N, making this architecture impractical beyond small grids.

---

## Roadmap — Version 2

Version 2 is under active local development and is **not yet committed to this repository**. Planned extensions include:

- **Multi-Agent Reinforcement Learning (MARL)** — Decentralized per-intersection agents, reducing per-agent action space from 32 to 2
- **Graph Neural Networks (GNNs)** — Graph Convolutional layers to process spatial intersection relationships natively
- **Shared Decentralized Policy** — A single weight set executed independently at each node
- **Neighbor Communication** — Augmenting local observations with encoded hidden states from adjacent intersections
- **Graph-Based State Representation** — Replacing the flat 25-feature vector with structured node and edge attributes
=======
### 1. Run Tests
```bash
# Agent / GNN / PPO gradient-flow tests (20 tests)
python -m pytest tests/test_agents.py -v

# Full Phase 3 environment validation (29 tests)
python -m pytest tests/test_phase3_e2e.py -v --tb=short

# Scripted signal-control validation
python tests/scripted_control_test.py
```

### 2. Training (EXP-06 — 150k steps, already completed)
```bash
python training/train_marl.py --total-steps 150000 --exp-name EXP-06
```

### 3. Run Final Benchmark (180 episodes)
```bash
python evaluation/phase4_benchmark.py \
    --episodes 10 \
    --sim-horizon 1200 \
    --model-dir results/EXP-06 \
    --out-dir results/phase4
```

### 4. Generate Plots
```bash
python evaluation/plot_phase4.py
```

### 5. GUI Demonstration (RL-controlled)
```bash
python scripts/demo_rl_gui.py \
    --sumo-cfg networks/grid_balanced.sumocfg \
    --sim-horizon 1200
```

The demo loads EXP-06, starts SUMO-GUI, and controls all 5 traffic signals via real RL inference. Phase switches are printed to the terminal as textual evidence.

---

## Project Structure

```
├── agents/
│   ├── gnn_encoder.py        # 2-layer GraphSAGE
│   └── ppo_policy.py         # Parameter-shared PPO (actor + critic)
├── communication/
│   └── graph_builder.py      # Builds PyG graph from observations
├── environment/
│   ├── multi_agent_env.py    # TrafficMARLEnv (PettingZoo ParallelEnv)
│   ├── builders.py           # ObsBuilder, RewardBuilder
│   └── interfaces.py         # Abstract interfaces
├── evaluation/
│   ├── phase4_benchmark.py   # 3×6×10 = 180-episode benchmark
│   ├── plot_phase4.py        # Visualization script
│   └── metrics_collector.py  # Episode metrics
├── networks/
│   ├── grid.net.xml          # 5-intersection SUMO network
│   ├── grid_balanced.sumocfg # Balanced traffic scenario
│   ├── grid_ns_heavy.sumocfg # N/S heavy scenario
│   └── ...                   # Other 4 scenarios
├── scripts/
│   └── demo_rl_gui.py        # Final GUI demonstration
├── tests/
│   ├── test_agents.py        # 20 unit tests (GNN, PPO, signal control)
│   └── test_phase3_e2e.py    # 29 integration tests
├── training/
│   └── train_marl.py         # Training script
├── results/
│   ├── EXP-06/               # Final trained model (150k steps)
│   │   ├── gnn_encoder.pt
│   │   └── ppo_policy.pt
│   └── phase4/               # Benchmark results
│       ├── raw_episodes.csv
│       ├── summary.csv
│       ├── timing_stats.csv
│       └── plots/
└── LOG.md                    # Engineering history
```

---

## Results Location
- **Trained models**: `results/EXP-06/`
- **Benchmark CSVs**: `results/phase4/`
- **Benchmark plots**: `results/phase4/plots/`
- **Final report**: `results/final_v2_report.md`

---

## Key Design Decisions

### GNN Gradient-Flow Fix (Phase 1)
The original implementation stored GNN embeddings during rollout under `torch.no_grad()`, disconnecting the GNN from the PPO gradient path. The fix re-constructs the graph and re-runs GNN inference during each PPO update step, ensuring full gradient flow through the entire pipeline.

### Signal Control Design
- `setPhase()` is called every 5s to prevent SUMO from auto-advancing its XML program
- `setPhaseDuration()` is set to `max_green - time_in_phase` so the GUI timer reflects real RL-controlled remaining time
- Yellow transitions (3s) are applied on phase switches
- Min/max green safety constraints are enforced by the environment, not the RL agent

### Benchmark Fairness
All controllers (Fixed-Time, Random, RL) run for the same **1200-second simulation horizon** — not the same number of loop iterations — ensuring fair comparison. Throughput is measured using `len(traci.simulation.getArrivedIDList())` accumulated per step.

---

## Limitations
- Training is CPU-only (no GPU required or used)
- 150k steps is relatively short for complex MARL convergence
- All 5 agents share a single policy network (parameter sharing)
- The network is a fixed 5-intersection 3×3 grid subset
>>>>>>> Stashed changes

---

## License
<<<<<<< Updated upstream

[MIT](LICENSE) © 2026
=======
MIT
>>>>>>> Stashed changes

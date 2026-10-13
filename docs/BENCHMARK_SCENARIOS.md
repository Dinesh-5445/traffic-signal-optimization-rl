# Traffic Optimization Benchmark Scenarios

This document describes the 6 benchmark scenarios used to evaluate the adaptive capabilities of the MARL traffic signal controllers.

## Scenarios

### 1. Balanced Traffic (Baseline)
- **Description:** Equal demand from all directions.
- **Goal:** Establishes baseline performance. Adaptive controllers should match or slightly exceed fixed-time controllers since demand is highly predictable and uniform.

### 2. North-South Heavy
- **Description:** Traffic flow is significantly higher on the North-South and South-North corridors. East-West traffic is reduced.
- **Goal:** Tests the controller's ability to allocate longer green phases to heavy traffic corridors, avoiding starvation on minor roads while maximizing throughput on major roads.

### 3. East-West Heavy
- **Description:** Opposite of North-South heavy. Increased East-West traffic, reduced North-South traffic.
- **Goal:** Ensures the controller isn't biased toward a specific corridor and can adapt dynamically to shifted dominant flows.

### 4. Rush Hour
- **Description:** Time-varying traffic demand.
  - 0–600 s: Balanced
  - 600–1200 s: North-South Heavy
  - 1200–1800 s: East-West Heavy
  - 1800–3600 s: Balanced
- **Goal:** Evaluates dynamic responsiveness. Controllers must adapt to shifting peak demands mid-simulation without getting stuck in outdated phase allocations.

### 5. Random Arrival Pattern
- **Description:** Replaces perfectly periodic vehicle arrivals with stochastic (probabilistic) generation.
- **Goal:** Tests robustness to variance and micro-level demand fluctuations (platooning vs. sparse arrivals), a key advantage of responsive RL models over fixed-time controllers.

### 6. Congestion Scenario
- **Description:** Vehicle generation deliberately exceeds the network's processing capacity.
- **Goal:** Evaluates queue management under saturation. Measures how gracefully the system degrades, specifically monitoring queue length, waiting times, and vehicle teleports (deadlocks).

## Running the Benchmark

To evaluate trained models across these scenarios:
```bash
python evaluation/benchmark_scenarios.py --episodes 1 --max-steps 1200
```
This script will produce `results/benchmarks/benchmark_results.csv`.

To generate comparison plots:
```bash
python evaluation/plot_benchmark.py
```
Plots will be saved in `results/benchmarks/plots/`.

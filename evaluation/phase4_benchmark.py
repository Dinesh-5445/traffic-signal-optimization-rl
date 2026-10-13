"""
evaluation/phase4_benchmark.py — Phase 4 Rigorous Benchmark

Runs 3 controllers × 6 scenarios × 10 episodes.

Controllers:
  - fixed_time:  SUMO's built-in fixed-time XML programme (no TraCI override)
  - random:      Random valid action per step (respects same env safety constraints)
  - rl_marl_gnn: Trained V2 PPO + GraphSAGE (EXP-06)

Scenarios:
  balanced, ns_heavy, ew_heavy, rush_hour, congested, random

Outputs (all in results/phase4/):
  raw_episodes.csv       — one row per (scenario, controller, episode)
  summary.csv            — mean ± std per (scenario, controller)
  timing_stats.csv       — RL signal timing distribution per intersection
  learning_curve.txt     — training log summary (extracted from EXP-06 output)

Usage:
  python evaluation/phase4_benchmark.py --episodes 10 --sim-horizon 1200
  python evaluation/phase4_benchmark.py --episodes 10 --sim-horizon 1200 --model-dir results/EXP-06
"""

import os, sys, csv, json, time, random, argparse
import numpy as np
import torch

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
import traci

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gnn_encoder import GNNEncoder
from agents.ppo_policy import ParameterSharedPPOPolicy
from environment.builders import ObsBuilder, MultiObjectiveRewardBuilder
from environment.multi_agent_env import TrafficMARLEnv
from communication.graph_builder import GraphBuilder

# ── Topology ────────────────────────────────────────────────────────────────
TS_IDS = ['A1', 'B0', 'B1', 'B2', 'C1']
TL_EDGES = {
    'A1': {'N': 'A2A1', 'S': 'A0A1', 'E': 'B1A1', 'W': None},
    'B0': {'N': 'B1B0', 'S': None,   'E': 'C0B0', 'W': 'A0B0'},
    'B1': {'N': 'B2B1', 'S': 'B0B1', 'E': 'C1B1', 'W': 'A1B1'},
    'B2': {'N': None,   'S': 'B1B2', 'E': 'C2B2', 'W': 'A2B2'},
    'C1': {'N': 'C2C1', 'S': 'C0C1', 'E': None,   'W': 'B1C1'},
}
NEIGHBOR_MAP = {
    'A1': ['B1'],
    'B0': ['B1'],
    'B1': ['A1', 'B0', 'B2', 'C1'],
    'B2': ['B1'],
    'C1': ['B1'],
}

SCENARIOS = {
    "balanced":  "networks/grid_balanced.sumocfg",
    "ns_heavy":  "networks/grid_ns_heavy.sumocfg",
    "ew_heavy":  "networks/grid_ew_heavy.sumocfg",
    "rush_hour": "networks/grid_rush_hour.sumocfg",
    "congested": "networks/grid_congested.sumocfg",
    "random":    "networks/grid_random.sumocfg",
}

NODE_FEAT_DIM = 9
GNN_OUT = 64

# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_queue(edge_id):
    if edge_id is None: return 0.0
    return float(traci.edge.getLastStepHaltingNumber(edge_id))

def _get_wait(edge_id):
    if edge_id is None: return 0.0
    return float(traci.edge.getWaitingTime(edge_id))

def _get_vehicle_count(edge_id):
    if edge_id is None: return 0
    return traci.edge.getLastStepVehicleNumber(edge_id)

def collect_step_metrics(ts_ids, tl_edges):
    """Return per-step aggregate metrics across all intersections."""
    total_queue = 0.0
    total_wait  = 0.0
    for tl in ts_ids:
        edges = tl_edges[tl]
        total_queue += sum(_get_queue(edges[d]) for d in ['N','S','E','W'])
        total_wait  += sum(_get_wait(edges[d])  for d in ['N','S','E','W'])
    arrived = len(traci.simulation.getArrivedIDList())   # true trip completions this step
    return total_queue / len(ts_ids), total_wait / len(ts_ids), arrived


# ── Fixed-Time controller ────────────────────────────────────────────────────

def run_fixed_time(sumo_cfg, sim_horizon, seed):
    """Run SUMO with NO TraCI override — the XML fixed-time programme controls signals."""
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "--no-step-log",
                f"--seed={seed}"]
    if traci.isLoaded():
        traci.close()
    traci.start(sumo_cmd)

    queues, waits, arrivals = [], [], []
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < sim_horizon:
        traci.simulationStep()
        q, w, a = collect_step_metrics(TS_IDS, TL_EDGES)
        queues.append(q); waits.append(w); arrivals.append(a)
        step += 1

    traci.close()
    return {
        "mean_queue": float(np.mean(queues)) if queues else 0.0,
        "mean_wait":  float(np.mean(waits))  if waits  else 0.0,
        "total_flow": int(sum(arrivals)),
        "steps":      step,
    }


# ── Random controller ────────────────────────────────────────────────────────

def run_random(sumo_cfg, sim_horizon, seed):
    """Random valid actions via the same TrafficMARLEnv (same safety constraints as RL)."""
    rng = random.Random(seed)
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "--no-step-log",
                f"--seed={seed}"]
    obs_builder    = ObsBuilder(TS_IDS, TL_EDGES)
    reward_builder = MultiObjectiveRewardBuilder(TS_IDS, TL_EDGES, NEIGHBOR_MAP)
    graph_builder  = GraphBuilder(TS_IDS, TL_EDGES)

    # Convert simulation horizon in seconds to environment steps (delta_time = 5 by default)
    env_max_steps = int(sim_horizon / 5)

    env = TrafficMARLEnv(
        sumo_cmd=sumo_cmd,
        ts_ids=TS_IDS,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        graph_builder=graph_builder,
        max_steps=env_max_steps,
    )

    obs, _ = env.reset()
    queues, waits, arrivals = [], [], []

    while env.agents:
        if not obs: break
        actions = {tl: rng.randint(0, 1) for tl in TS_IDS}
        obs, _, term, trunc, _ = env.step(actions)
        q, w, _ = collect_step_metrics(TS_IDS, TL_EDGES)  # queue+wait only; arrivals captured inside env
        a = env._step_arrived   # correct: accumulated across every internal simulationStep()
        queues.append(q); waits.append(w); arrivals.append(a)
        if all(term.get(tl, False) or trunc.get(tl, False) for tl in TS_IDS):
            break

    env.close()
    return {
        "mean_queue": float(np.mean(queues)) if queues else 0.0,
        "mean_wait":  float(np.mean(waits))  if waits  else 0.0,
        "total_flow": int(sum(arrivals)),
        "steps":      len(queues),
    }


# ── RL controller ────────────────────────────────────────────────────────────

def load_rl_models(model_dir, device):
    gnn    = GNNEncoder(NODE_FEAT_DIM, 64, GNN_OUT).to(device)
    policy = ParameterSharedPPOPolicy(GNN_OUT, action_dim=2, hidden_dim=64).to(device)
    gnn.load_state_dict(torch.load(
        os.path.join(model_dir, "gnn_encoder.pt"), map_location=device, weights_only=True))
    policy.load_state_dict(torch.load(
        os.path.join(model_dir, "ppo_policy.pt"),  map_location=device, weights_only=True))
    gnn.eval(); policy.eval()
    return gnn, policy


def run_rl(sumo_cfg, sim_horizon, seed, gnn, policy, device, collect_timing=False):
    """Run the trained RL policy. No gradient updates. Fresh episode per call."""
    torch.manual_seed(seed)
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "--no-step-log",
                f"--seed={seed}"]
    obs_builder    = ObsBuilder(TS_IDS, TL_EDGES)
    reward_builder = MultiObjectiveRewardBuilder(TS_IDS, TL_EDGES, NEIGHBOR_MAP)
    graph_builder  = GraphBuilder(TS_IDS, TL_EDGES)

    # Convert simulation horizon in seconds to environment steps (delta_time = 5 by default)
    env_max_steps = int(sim_horizon / 5)

    env = TrafficMARLEnv(
        sumo_cmd=sumo_cmd,
        ts_ids=TS_IDS,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        graph_builder=graph_builder,
        max_steps=env_max_steps,
    )

    obs, _ = env.reset()
    queues, waits, arrivals = [], [], []

    # For timing stats: track green durations per tl
    green_starts  = {tl: None for tl in TS_IDS}
    green_durations = {tl: [] for tl in TS_IDS}
    switch_counts   = {tl: 0  for tl in TS_IDS}
    last_phase      = {tl: None for tl in TS_IDS}

    while env.agents:
        if not obs: break

        graph = graph_builder.build_graph(env_state=None, obs=obs).to(device)
        with torch.no_grad():
            embeddings         = gnn(graph)
            actions_t, _, _, _ = policy.get_action_and_value(embeddings)
        actions_np  = actions_t.cpu().numpy()
        action_dict = {ts: int(actions_np[i]) for i, ts in enumerate(TS_IDS)}

        obs, _, term, trunc, _ = env.step(action_dict)
        q, w, _ = collect_step_metrics(TS_IDS, TL_EDGES)  # queue+wait only; arrivals captured inside env
        a = env._step_arrived   # correct: accumulated across every internal simulationStep()
        queues.append(q); waits.append(w); arrivals.append(a)

        # Timing tracking
        if collect_timing:
            sim_time = traci.simulation.getTime()
            for tl in TS_IDS:
                ph = traci.trafficlight.getPhase(tl)
                if ph in [0, 2]:  # green phases only
                    if last_phase[tl] != ph:
                        # New green phase started
                        if last_phase[tl] in [0, 2] and green_starts[tl] is not None:
                            dur = sim_time - green_starts[tl]
                            green_durations[tl].append(dur)
                            switch_counts[tl] += 1
                        green_starts[tl] = sim_time
                last_phase[tl] = ph

        if all(term.get(tl, False) or trunc.get(tl, False) for tl in TS_IDS):
            break

    env.close()

    timing = {}
    if collect_timing:
        for tl in TS_IDS:
            durs = green_durations[tl]
            if durs:
                timing[tl] = {
                    "min_green":    float(np.min(durs)),
                    "max_green":    float(np.max(durs)),
                    "mean_green":   float(np.mean(durs)),
                    "median_green": float(np.median(durs)),
                    "std_green":    float(np.std(durs)),
                    "switches":     switch_counts[tl],
                    "n_greens":     len(durs),
                }
            else:
                timing[tl] = {"min_green": 0, "max_green": 0, "mean_green": 0,
                               "median_green": 0, "std_green": 0, "switches": 0, "n_greens": 0}

    return {
        "mean_queue": float(np.mean(queues)) if queues else 0.0,
        "mean_wait":  float(np.mean(waits))  if waits  else 0.0,
        "total_flow": int(sum(arrivals)),
        "steps":      len(queues),
        "timing":     timing,
    }


# ── Main benchmark loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--sim-horizon",type=int, default=1200)
    parser.add_argument("--model-dir",  default="results/EXP-06")
    parser.add_argument("--out-dir",    default="results/phase4_final")
    parser.add_argument("--seed-start", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 4 Benchmark | device={device} | episodes={args.episodes} | sim_horizon={args.sim_horizon}s")

    # Load RL models once
    print(f"\nLoading RL models from {args.model_dir}...")
    if not os.path.exists(os.path.join(args.model_dir, "gnn_encoder.pt")):
        print("ERROR: RL models not found. Run EXP-06 training first.")
        sys.exit(1)
    gnn, policy = load_rl_models(args.model_dir, device)
    print("RL models loaded.")

    # Output files
    raw_path     = os.path.join(args.out_dir, "raw_episodes.csv")
    summary_path = os.path.join(args.out_dir, "summary.csv")
    timing_path  = os.path.join(args.out_dir, "timing_stats.csv")

    raw_fieldnames = [
        "scenario", "controller", "episode", "seed",
        "mean_queue", "mean_wait", "total_flow", "steps"
    ]
    timing_fieldnames = [
        "scenario", "episode", "intersection",
        "min_green", "max_green", "mean_green", "median_green",
        "std_green", "switches", "n_greens"
    ]

    raw_rows     = []
    timing_rows  = []

    controllers = ["fixed_time", "random", "rl_marl_gnn"]

    for scenario_name, sumo_cfg in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario_name}")
        print(f"{'='*60}")

        for controller in controllers:
            print(f"\n  Controller: {controller}")
            ep_results = []

            for ep in range(args.episodes):
                seed = args.seed_start + ep * 7  # well-spaced seeds

                t0 = time.time()
                try:
                    if controller == "fixed_time":
                        result = run_fixed_time(sumo_cfg, args.sim_horizon, seed)
                    elif controller == "random":
                        result = run_random(sumo_cfg, args.sim_horizon, seed)
                    else:  # rl_marl_gnn
                        # Collect timing stats on episode 0 of each scenario
                        result = run_rl(sumo_cfg, args.sim_horizon, seed, gnn, policy, device,
                                        collect_timing=(ep == 0))
                        if ep == 0 and result["timing"]:
                            for tl, t in result["timing"].items():
                                timing_rows.append({
                                    "scenario": scenario_name,
                                    "episode": ep + 1,
                                    "intersection": tl,
                                    **t,
                                })
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"    Episode {ep+1} FAILED: {e}")
                    result = {"mean_queue": None, "mean_wait": None,
                              "total_flow": None, "steps": 0}

                elapsed = time.time() - t0
                ep_results.append(result)

                row = {
                    "scenario":    scenario_name,
                    "controller":  controller,
                    "episode":     ep + 1,
                    "seed":        seed,
                    "mean_queue":  result["mean_queue"],
                    "mean_wait":   result["mean_wait"],
                    "total_flow":  result["total_flow"],
                    "steps":       result["steps"],
                }
                raw_rows.append(row)

                print(
                    f"    Ep {ep+1:>2}/{args.episodes} | seed={seed} | "
                    f"Queue={result['mean_queue']:>6.2f} | "
                    f"Wait={result['mean_wait']:>7.2f}s | "
                    f"Flow={result['total_flow']:>4} | "
                    f"Steps={result['steps']:>4} | {elapsed:.1f}s"
                )

            # Per-controller summary across episodes
            valid = [r for r in ep_results if r["mean_queue"] is not None]
            if valid:
                print(
                    f"  >> {controller}: "
                    f"Queue={np.mean([r['mean_queue'] for r in valid]):.3f}"
                    f"±{np.std([r['mean_queue'] for r in valid]):.3f} | "
                    f"Wait={np.mean([r['mean_wait'] for r in valid]):.3f}"
                    f"±{np.std([r['mean_wait'] for r in valid]):.3f} | "
                    f"Flow={np.mean([r['total_flow'] for r in valid]):.1f}"
                    f"±{np.std([r['total_flow'] for r in valid]):.1f}"
                )

    # -- Save raw episodes ----------------------------------------------------
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fieldnames)
        w.writeheader()
        w.writerows(raw_rows)
    print(f"\nRaw episodes saved -> {raw_path}")

    # -- Save timing stats ----------------------------------------------------
    if timing_rows:
        with open(timing_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=timing_fieldnames)
            w.writeheader()
            w.writerows(timing_rows)
        print(f"Timing stats saved -> {timing_path}")

    # -- Compute and save summary ---------------------------------------------
    summary_rows = []
    # Group raw_rows by (scenario, controller)
    from itertools import groupby
    from operator import itemgetter

    sorted_rows = sorted(raw_rows, key=itemgetter("scenario", "controller"))
    for (scen, ctrl), group in groupby(sorted_rows, key=lambda r: (r["scenario"], r["controller"])):
        g = list(group)
        queues = [r["mean_queue"] for r in g if r["mean_queue"] is not None]
        waits  = [r["mean_wait"]  for r in g if r["mean_wait"]  is not None]
        flows  = [r["total_flow"] for r in g if r["total_flow"] is not None]
        summary_rows.append({
            "scenario":       scen,
            "controller":     ctrl,
            "episodes":       len(g),
            "queue_mean":     np.mean(queues) if queues else None,
            "queue_std":      np.std(queues)  if queues else None,
            "wait_mean":      np.mean(waits)  if waits  else None,
            "wait_std":       np.std(waits)   if waits  else None,
            "flow_mean":      np.mean(flows)  if flows  else None,
            "flow_std":       np.std(flows)   if flows  else None,
        })

    summary_fieldnames = [
        "scenario","controller","episodes",
        "queue_mean","queue_std","wait_mean","wait_std",
        "flow_mean","flow_std",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary saved -> {summary_path}")

    # ── Print comparison tables ──────────────────────────────────────────────
    print("\n" + "="*90)
    print("  PHASE 4 COMPARISON TABLE (Mean ± Std across episodes)")
    print("="*90)
    print(f"{'Scenario':<12} {'Controller':<14} {'Queue Mean':>10} {'Queue Std':>10} "
          f"{'Wait Mean':>10} {'Wait Std':>10} {'Flow Mean':>10} {'Flow Std':>10}")
    print("-"*90)
    for r in summary_rows:
        def fmt(v): return f"{v:>10.3f}" if v is not None else f"{'N/A':>10}"
        print(
            f"{r['scenario']:<12} {r['controller']:<14} "
            f"{fmt(r['queue_mean'])} {fmt(r['queue_std'])} "
            f"{fmt(r['wait_mean'])} {fmt(r['wait_std'])} "
            f"{fmt(r['flow_mean'])} {fmt(r['flow_std'])}"
        )

    # ── Compute improvement % over Fixed-Time ────────────────────────────────
    print("\n" + "="*90)
    print("  IMPROVEMENT vs FIXED-TIME (lower-is-better for queue/wait; higher for flow)")
    print("="*90)
    print(f"{'Scenario':<12} {'Controller':<14} {'Queue Delta%':>10} {'Wait Delta%':>10} {'Flow Delta%':>10}")
    print("-"*50)

    # Index summary by (scenario, controller)
    sum_idx = {(r["scenario"], r["controller"]): r for r in summary_rows}

    for scen in SCENARIOS:
        ft = sum_idx.get((scen, "fixed_time"))
        for ctrl in ["random", "rl_marl_gnn"]:
            cr = sum_idx.get((scen, ctrl))
            if ft and cr:
                def pct_lower(base, new):
                    if base and base > 0: return (base - new) / base * 100
                    return None
                def pct_higher(base, new):
                    if base and base > 0: return (new - base) / base * 100
                    return None
                dq = pct_lower(ft["queue_mean"], cr["queue_mean"])
                dw = pct_lower(ft["wait_mean"],  cr["wait_mean"])
                df = pct_higher(ft["flow_mean"], cr["flow_mean"])
                def fmtd(v): return f"{v:>+10.1f}%" if v is not None else f"{'N/A':>10}"
                print(f"{scen:<12} {ctrl:<14} {fmtd(dq)} {fmtd(dw)} {fmtd(df)}")

    print(f"\nAll outputs written to: {args.out_dir}/")


if __name__ == "__main__":
    main()

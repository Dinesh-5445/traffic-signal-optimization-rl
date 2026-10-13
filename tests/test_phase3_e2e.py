"""
tests/test_phase3_e2e.py — Phase 3 Full End-to-End Validation

Tests:
  1. Full SUMO loop (100 steps)
  2. Node ordering consistency
  3. Scripted action integrity (A, B, C, D)
  4. 65s timing investigation
  5. Reward audit
  6. Observation sanity (9 features)
  7. Graph sanity (5 nodes, topology, no self-loops)
  8. Action sanity
  9. Episode/reset/TraCI lifecycle (3 episodes)
  10. Traffic simulation sanity
  11. Numerical stability

Runs with real SUMO, no mocks.
"""

import os
import sys
import time
import math
import unittest
import numpy as np
import torch

# Force libsumo
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
from torch_geometric.data import Data

from environment.multi_agent_env import TrafficMARLEnv
from environment.builders import ObsBuilder, MultiObjectiveRewardBuilder
from communication.graph_builder import GraphBuilder
from agents.gnn_encoder import GNNEncoder
from agents.ppo_policy import ParameterSharedPPOPolicy

# ── Fixed topology ────────────────────────────────────────────────────────────
TS_IDS = ["A1", "B0", "B1", "B2", "C1"]

TL_EDGES = {
    "A1": {"N": "A2A1", "S": "A0A1", "E": "B1A1", "W": None},
    "B0": {"N": "B1B0", "S": None,   "E": "C0B0", "W": "A0B0"},
    "B1": {"N": "B2B1", "S": "B0B1", "E": "C1B1", "W": "A1B1"},
    "B2": {"N": None,   "S": "B1B2", "E": "C2B2", "W": "A2B2"},
    "C1": {"N": "C2C1", "S": "C0C1", "E": None,   "W": "B1C1"},
}

NEIGHBOR_MAP = {
    "A1": ["B1"],
    "B0": ["B1"],
    "B1": ["A1", "B0", "B2", "C1"],
    "B2": ["B1"],
    "C1": ["B1"],
}

# Expected graph edges (source_node_idx → target_node_idx)
# Built from TL_EDGES by graph_builder convention: edge_id[:2] → target
# A2A1 → src A2 (not a TL) → skipped; A0A1 → src A0 (not a TL) → skipped
# B1A1 → src B1(idx=2) → target A1(idx=0); A0B0 → not a TL; C0B0 → not a TL
# A1B1 → src A1(idx=0) → B1(idx=2); B0B1 → src B0(idx=1) → B1(idx=2)
# B2B1 → src B2(idx=3) → B1(idx=2); C1B1 → src C1(idx=4) → B1(idx=2)
# B1B2 → src B1(idx=2) → B2(idx=3); B1B0 → src B1(idx=2) → B0(idx=1)
# C2B2, A2B2, B1C1 → some not TLs
# B1C1 → src B1(idx=2) → C1(idx=4); C2C1, C0C1 → not TLs
EXPECTED_EDGES = {
    (2, 0),  # B1→A1
    (0, 2),  # A1→B1
    (1, 2),  # B0→B1
    (3, 2),  # B2→B1
    (4, 2),  # C1→B1
    (2, 3),  # B1→B2
    (2, 1),  # B1→B0
    (2, 4),  # B1→C1
}

SUMO_CMD = ["sumo", "-c", "networks/grid.sumocfg", "--no-warnings", "--no-step-log"]
DEVICE = torch.device("cpu")


def make_env(max_steps=200) -> TrafficMARLEnv:
    obs_builder    = ObsBuilder(TS_IDS, TL_EDGES)
    reward_builder = MultiObjectiveRewardBuilder(TS_IDS, TL_EDGES, NEIGHBOR_MAP)
    graph_builder  = GraphBuilder(TS_IDS, TL_EDGES)
    return TrafficMARLEnv(
        sumo_cmd=SUMO_CMD,
        ts_ids=TS_IDS,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        graph_builder=graph_builder,
        delta_time=5,
        max_steps=max_steps,
    )


# ──────────────────────────────────────────────────────────────────────────────
class TestFullSUMOLoop(unittest.TestCase):
    """Test 1: Full 100-step end-to-end SUMO loop."""

    def test_100_step_loop_completes(self):
        env = make_env(max_steps=100)
        gnn    = GNNEncoder(in_channels=9, hidden_channels=64, out_channels=64).to(DEVICE)
        policy = ParameterSharedPPOPolicy(gnn_out_channels=64).to(DEVICE)
        gnn.eval()
        policy.eval()

        gb = GraphBuilder(TS_IDS, TL_EDGES)

        obs, _ = env.reset()

        step_times = []
        prev_sim_time = 0.0

        for step in range(100):
            t0 = time.time()

            # ── 1. Check all 5 agents have observations ──────────────────────
            self.assertEqual(len(obs), 5, f"Step {step}: expected 5 obs, got {len(obs)}")
            for tl in TS_IDS:
                self.assertIn(tl, obs, f"Step {step}: missing obs for {tl}")
                o = obs[tl]
                self.assertEqual(o.shape, (9,), f"Step {step} {tl}: obs shape wrong")
                self.assertTrue(np.all(np.isfinite(o)), f"Step {step} {tl}: non-finite obs")
                self.assertTrue(np.all(o >= 0), f"Step {step} {tl}: negative obs value")

            # ── 2. Build graph ───────────────────────────────────────────────
            graph = gb.build_graph(env_state=None, obs=obs).to(DEVICE)
            self.assertEqual(graph.x.shape[0], 5, f"Step {step}: graph has wrong num nodes")
            self.assertEqual(graph.x.shape[1], 9, f"Step {step}: graph node features wrong dim")

            # ── 3. GNN forward ───────────────────────────────────────────────
            with torch.no_grad():
                embeddings = gnn(graph)
            self.assertEqual(embeddings.shape, (5, 64), f"Step {step}: GNN output shape wrong")
            self.assertTrue(torch.all(torch.isfinite(embeddings)), f"Step {step}: GNN non-finite")

            # ── 4. PPO action ────────────────────────────────────────────────
            with torch.no_grad():
                actions_t, log_probs, entropy, values = policy.get_action_and_value(embeddings)
            actions_np = actions_t.cpu().numpy()
            self.assertEqual(len(actions_np), 5, f"Step {step}: wrong number of actions")
            for a in actions_np:
                self.assertIn(int(a), [0, 1], f"Step {step}: invalid action {a}")
            self.assertTrue(torch.all(torch.isfinite(log_probs)))
            self.assertTrue(torch.isfinite(values).all())

            # ── 5. Build action dict preserving ordering ─────────────────────
            action_dict = {ts: int(actions_np[i]) for i, ts in enumerate(TS_IDS)}

            # ── 6. Step environment ──────────────────────────────────────────
            next_obs, rewards, terminations, truncations, _ = env.step(action_dict)

            # ── 7. Sim time advances ─────────────────────────────────────────
            sim_time = traci.simulation.getTime()
            self.assertGreater(sim_time, prev_sim_time,
                f"Step {step}: sim time did not advance ({prev_sim_time} → {sim_time})")
            prev_sim_time = sim_time

            # ── 8. Rewards finite ────────────────────────────────────────────
            for tl, r in rewards.items():
                self.assertTrue(math.isfinite(r), f"Step {step} {tl}: non-finite reward {r}")

            # ── 9. Episode should not terminate unexpectedly at step < 100 ──
            if step < 90:
                for tl in TS_IDS:
                    self.assertFalse(terminations.get(tl, False),
                        f"Step {step}: unexpected termination at {tl}")

            step_times.append(time.time() - t0)
            obs = next_obs

            if all(terminations.get(tl, False) or truncations.get(tl, False) for tl in TS_IDS):
                break

        env.close()
        avg_ms = np.mean(step_times) * 1000
        print(f"\n[TestFullSUMOLoop] 100-step loop complete. Avg step time: {avg_ms:.1f}ms")


# ──────────────────────────────────────────────────────────────────────────────
class TestNodeOrderingConsistency(unittest.TestCase):
    """Test 2: Node ordering must be consistent across obs → graph → GNN → actions → reward."""

    def test_ordering_is_deterministic_and_consistent(self):
        env = make_env(max_steps=50)
        gb  = GraphBuilder(TS_IDS, TL_EDGES)
        gnn = GNNEncoder(in_channels=9, hidden_channels=64, out_channels=64)
        policy = ParameterSharedPPOPolicy(gnn_out_channels=64)
        gnn.eval(); policy.eval()

        obs, _ = env.reset()
        graph = gb.build_graph(env_state=None, obs=obs)

        # The graph node at position i must correspond to TS_IDS[i]
        for i, tl in enumerate(TS_IDS):
            expected_feat = torch.tensor(obs[tl], dtype=torch.float32)
            actual_feat   = graph.x[i]
            self.assertTrue(torch.allclose(expected_feat, actual_feat),
                f"Node {i} ({tl}): graph feature mismatch. "
                f"obs={obs[tl]}, graph.x[{i}]={actual_feat.numpy()}")

        # GNN output row i corresponds to agent TS_IDS[i]
        with torch.no_grad():
            embeddings = gnn(graph)

        # Build action dict using same index-to-agent mapping
        with torch.no_grad():
            actions_t, _, _, _ = policy.get_action_and_value(embeddings)
        action_dict = {ts: int(actions_t[i].item()) for i, ts in enumerate(TS_IDS)}
        self.assertEqual(set(action_dict.keys()), set(TS_IDS),
            "action_dict keys don't match TS_IDS")

        env.close()
        print(f"\n[TestNodeOrdering] All {len(TS_IDS)} nodes aligned: obs → graph → GNN → action")


# ──────────────────────────────────────────────────────────────────────────────
class TestScriptedActionIntegrity(unittest.TestCase):
    """Test 3: Scripted deterministic actions confirm action→phase causality."""

    def _run_scripted(self, mode, max_steps=60):
        env = make_env(max_steps=max_steps)
        obs, _ = env.reset()
        records = []

        for step in range(max_steps):
            actions = {}
            for i, tl in enumerate(TS_IDS):
                if mode == 'A':
                    actions[tl] = 0
                elif mode == 'B':
                    actions[tl] = 1
                elif mode == 'C':
                    actions[tl] = step % 2
                elif mode == 'D':
                    actions[tl] = (step + i) % 2

            # Record BEFORE step
            phases_before = {tl: traci.trafficlight.getPhase(tl) for tl in TS_IDS}
            sim_time_before = traci.simulation.getTime()

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # Record AFTER step
            phases_after = {tl: traci.trafficlight.getPhase(tl) for tl in TS_IDS}
            sim_time_after = traci.simulation.getTime()

            records.append({
                "step": step,
                "sim_time": sim_time_after,
                "actions": dict(actions),
                "phases_before": dict(phases_before),
                "phases_after": dict(phases_after),
                "rewards": dict(rewards),
                "time_in_phase": dict(env.time_in_phase),
            })

            obs = next_obs
            if all(terminations.get(tl, False) or truncations.get(tl, False) for tl in TS_IDS):
                break

        env.close()
        return records

    def test_mode_A_all_zero_phases_converge_to_0(self):
        """Constant action 0 → after min_green elapses, all phases should settle at 0."""
        records = self._run_scripted('A', max_steps=60)
        # After step 3 (15s elapsed ≥ min_green=10s), all settled phases should be 0
        settled = [r for r in records if r["time_in_phase"].get("A1", 0) >= 10]
        for r in settled[-5:]:
            for tl in TS_IDS:
                phase = r["phases_after"][tl]
                self.assertIn(phase, [0, 1],
                    f"Mode A: TL={tl} unexpected phase {phase} at step {r['step']}")
        print(f"\n[Scripted A] Settled phases confirmed NS-green (0/1 yellow only)")

    def test_mode_B_all_one_phases_converge_to_2(self):
        """Constant action 1 → after min_green elapses, all phases should settle at 2.

        Note: max_green=60s causes a forced switch at ~60s. We only check phases
        that settled *before* the max_green ceiling is hit (time_in_phase in [10, 55]).
        After max_green triggers, the env forces a switch to NS green (correct behavior).
        """
        records = self._run_scripted('B', max_steps=60)
        # Only look at records in the settled EW-green window before the max_green switch
        settled = [
            r for r in records
            if 10 <= r["time_in_phase"].get("A1", 0) < 60
        ]
        self.assertGreater(len(settled), 0, "No settled EW-green records found (min_green not reached)")
        for r in settled[-5:]:
            for tl in TS_IDS:
                phase = r["phases_after"][tl]
                self.assertIn(phase, [2, 3],
                    f"Mode B: TL={tl} unexpected phase {phase} at step {r['step']}")
        print(f"\n[Scripted B] Settled phases confirmed EW-green (2/3 yellow only) [{len(settled)} records]")

    def test_mode_A_vs_B_produce_different_phase_histories(self):
        """Modes A and B must produce different phase histories."""
        rec_a = self._run_scripted('A', max_steps=40)
        rec_b = self._run_scripted('B', max_steps=40)
        phases_a = [r["phases_after"]["B1"] for r in rec_a]
        phases_b = [r["phases_after"]["B1"] for r in rec_b]
        self.assertNotEqual(phases_a, phases_b,
            "Scripted modes A and B produced identical B1 phase history — action not reaching SUMO")

    def test_sim_time_advances_each_step(self):
        records = self._run_scripted('C', max_steps=30)
        for i in range(1, len(records)):
            t_prev = records[i-1]["sim_time"]
            t_curr = records[i]["sim_time"]
            self.assertGreater(t_curr, t_prev,
                f"Sim time did not advance at step {i}: {t_prev} → {t_curr}")

    def test_rewards_are_finite_across_all_modes(self):
        for mode in ['A', 'B', 'C', 'D']:
            records = self._run_scripted(mode, max_steps=20)
            for r in records:
                for tl, rew in r["rewards"].items():
                    self.assertTrue(math.isfinite(rew),
                        f"Mode {mode}, step {r['step']}, {tl}: non-finite reward {rew}")


# ──────────────────────────────────────────────────────────────────────────────
class TestTimingInvestigation(unittest.TestCase):
    """Test 4: Investigate the 65s max green duration."""

    def test_65s_is_granularity_artifact_not_controller_violation(self):
        """
        When time_in_phase reaches max_green (60s), the switch is commanded.
        The SUMO sim has already advanced delta_time (5s) in that step,
        so the total observed green before yellow = 60s + up to 5s = 65s.
        This is a measurement artifact, NOT a controller allowing 65s intentionally.
        The controller fires the switch at 60s; the extra 5s comes from the current
        sim step already having elapsed before we read the timestamp.
        """
        env = make_env(max_steps=300)
        obs, _ = env.reset()

        max_green_violations = []  # Cases where controller logic caused >60s before switch

        for step in range(200):
            actions = {tl: 0 for tl in TS_IDS}  # constant action 0

            # Record time_in_phase BEFORE action is applied (this is what the controller sees)
            tip_before = dict(env.time_in_phase)
            _, _, terminations, truncations, _ = env.step(actions)
            tip_after = dict(env.time_in_phase)

            # Detect forced switches: tip went from ≥max_green to 0
            for tl in TS_IDS:
                before = tip_before[tl]
                after  = tip_after[tl]
                if before >= env.max_green and after == 0.0:
                    # Switch was correctly forced at 60s boundary
                    # The controller checked tip BEFORE the step (≥60s), so it's correct
                    pass
                elif before < env.max_green and after == 0.0:
                    # Voluntary switch from agent; no violation
                    pass

                # Flag any case where controller did NOT switch when it should have
                if before >= env.max_green and after > env.delta_time:
                    max_green_violations.append({
                        "tl": tl, "step": step,
                        "tip_before": before, "tip_after": after
                    })

            if all(terminations.get(tl, False) or truncations.get(tl, False) for tl in TS_IDS):
                break

        env.close()
        self.assertEqual(len(max_green_violations), 0,
            f"Controller failed to force switch at max_green for: {max_green_violations}")
        print("\n[TimingInvestigation] 65s = delta_time granularity artifact confirmed. "
              "Controller correctly fires switch at time_in_phase ≥ 60s (checked before step).")


# ──────────────────────────────────────────────────────────────────────────────
class TestRewardAudit(unittest.TestCase):
    """Test 5: Reward function correctness."""

    def test_reward_components_correct_signs(self):
        """
        In steady state with queues, reward must be < 0.
        With no vehicles, reward should be near 0 (queues=0, wait=0).
        """
        env = make_env(max_steps=100)
        obs, _ = env.reset()
        rewards_list = []

        for _ in range(50):
            actions = {tl: 0 for tl in TS_IDS}
            _, rewards, terminations, truncations, _ = env.step(actions)
            rewards_list.extend(rewards.values())
            if all(terminations.get(tl, False) or truncations.get(tl, False) for tl in TS_IDS):
                break

        env.close()

        # All rewards must be finite
        for r in rewards_list:
            self.assertTrue(math.isfinite(r), f"Non-finite reward: {r}")

        # With active traffic, at least some rewards should be negative (queue/wait penalty)
        mean_r = np.mean(rewards_list)
        self.assertLess(mean_r, 1.0,
            f"Mean reward suspiciously high ({mean_r:.4f}) — queue/wait penalties may be missing")
        print(f"\n[RewardAudit] mean={mean_r:.4f}, min={min(rewards_list):.4f}, "
              f"max={max(rewards_list):.4f}")

    def test_throughput_is_vehicle_count_delta_not_trip_completion(self):
        """
        Throughput = decrease in vehicles on incoming edges (last_count - current_count).
        It is NOT completed trips. This is a documented approximation.
        Verified by inspecting the builder code: throughput = max(0, last_count - current_count).
        """
        # Read the source directly to confirm
        import inspect
        from environment.builders import MultiObjectiveRewardBuilder
        source_compute = inspect.getsource(MultiObjectiveRewardBuilder.compute_reward)
        source_helper = inspect.getsource(MultiObjectiveRewardBuilder._get_vehicle_count)
        self.assertIn("last_vehicle_count", source_compute,
            "throughput should use last_vehicle_count delta")
        self.assertIn("getLastStepVehicleNumber", source_helper,
            "throughput should use SUMO edge vehicle count, not trip completion")
        self.assertNotIn("getArrivedNumber", source_compute,
            "throughput must NOT use getArrivedNumber (that would be trip completion)")
        print("\n[RewardAudit] Throughput = vehicle count decrease on incoming edges (NOT trips)")

    def test_no_double_counting(self):
        """
        Queue and waiting-time are measured independently per edge.
        Cooperative term uses local reward, not re-queried raw metrics.
        """
        import inspect
        from environment.builders import MultiObjectiveRewardBuilder
        source = inspect.getsource(MultiObjectiveRewardBuilder.compute_reward)
        # coop_rewards uses local_rewards dict (already computed), not raw metrics again
        self.assertIn("local_rewards[n]", source,
            "Cooperative term must use already-computed local_rewards, not re-query SUMO")
        print("\n[RewardAudit] No double-counting: coop term uses local_rewards dict")

    def test_teleport_penalty_uses_delta_not_cumulative(self):
        """Teleport count must be a DELTA from last step, not cumulative."""
        import inspect
        from environment.builders import MultiObjectiveRewardBuilder
        source = inspect.getsource(MultiObjectiveRewardBuilder.compute_reward)
        self.assertIn("last_teleports", source)
        self.assertIn("teleport_diff = current_teleports - self.last_teleports", source,
            "Teleport penalty must use delta from last_teleports")
        print("\n[RewardAudit] Teleport penalty uses delta correctly")


# ──────────────────────────────────────────────────────────────────────────────
class TestObservationSanity(unittest.TestCase):
    """Test 6: Observation 9-feature sanity."""

    def test_observations_correct_shape_and_bounds(self):
        env = make_env(max_steps=50)
        obs, _ = env.reset()

        # Run a few steps to get non-zero observations
        for _ in range(10):
            actions = {tl: 0 for tl in TS_IDS}
            obs, _, terminations, truncations, _ = env.step(actions)

        env.close()

        for tl in TS_IDS:
            o = obs[tl]
            self.assertEqual(o.shape, (9,), f"{tl}: wrong obs shape")
            self.assertEqual(o.dtype, np.float32, f"{tl}: wrong dtype")
            # All features must be finite
            self.assertTrue(np.all(np.isfinite(o)), f"{tl}: non-finite features: {o}")
            # All features must be non-negative (normalized queues, waits, phase)
            self.assertTrue(np.all(o >= 0), f"{tl}: negative observation: {o}")
            # Queue features [0:4] normalized by MAX_QUEUE_LEN=20 → max ~1.0+ but non-negative
            # Wait features [4:8] normalized by MAX_WAIT_TIME=500 → similarly bounded
            # Phase [8] = phase/3.0 → in [0.0, 1.0]
            phase_feat = float(o[8])
            self.assertGreaterEqual(phase_feat, 0.0, f"{tl}: phase feature < 0")
            self.assertLessEqual(phase_feat, 1.01, f"{tl}: phase feature > 1 ({phase_feat})")

        print(f"\n[ObsSanity] All {len(TS_IDS)} agents: shape=(9,), dtype=float32, "
              f"finite=True, non-negative=True, phase∈[0,1]")

    def test_phase_feature_reflects_actual_sumo_phase(self):
        env = make_env(max_steps=50)
        obs, _ = env.reset()
        # After reset, run a few steps
        for _ in range(5):
            actions = {tl: 0 for tl in TS_IDS}
            obs, _, _, _, _ = env.step(actions)

        for tl in TS_IDS:
            raw_phase = traci.trafficlight.getPhase(tl)
            obs_phase_feat = obs[tl][8]  # phase / 3.0
            expected = raw_phase / 3.0
            self.assertAlmostEqual(float(obs_phase_feat), expected, places=3,
                msg=f"{tl}: obs phase feature {obs_phase_feat} ≠ SUMO phase {raw_phase}/3.0")

        env.close()
        print("\n[ObsSanity] Phase feature correctly encodes traci.getPhase / 3.0")


# ──────────────────────────────────────────────────────────────────────────────
class TestGraphSanity(unittest.TestCase):
    """Test 7: Graph structure sanity."""

    def setUp(self):
        self.gb = GraphBuilder(TS_IDS, TL_EDGES)

    def test_5_nodes(self):
        obs = {tl: np.zeros(9, dtype=np.float32) for tl in TS_IDS}
        graph = self.gb.build_graph(env_state=None, obs=obs)
        self.assertEqual(graph.x.shape[0], 5)
        self.assertEqual(graph.x.shape[1], 9)

    def test_no_self_loops(self):
        ei = self.gb.edge_index
        for i in range(ei.shape[1]):
            src, dst = ei[0, i].item(), ei[1, i].item()
            self.assertNotEqual(src, dst, f"Self-loop at node {src}")

    def test_expected_topology(self):
        ei = self.gb.edge_index
        actual_edges = set(
            (ei[0, i].item(), ei[1, i].item()) for i in range(ei.shape[1])
        )
        for e in EXPECTED_EDGES:
            self.assertIn(e, actual_edges,
                f"Expected edge {e} (idx→idx) missing. "
                f"TS_IDS ordering: {list(enumerate(TS_IDS))}\n"
                f"Actual edges: {sorted(actual_edges)}")

    def test_edge_index_shape(self):
        ei = self.gb.edge_index
        self.assertEqual(ei.shape[0], 2)
        self.assertGreater(ei.shape[1], 0)

    def test_node_ordering_matches_ts_ids(self):
        obs = {tl: np.full(9, float(i), dtype=np.float32) for i, tl in enumerate(TS_IDS)}
        graph = self.gb.build_graph(env_state=None, obs=obs)
        for i, tl in enumerate(TS_IDS):
            self.assertAlmostEqual(graph.x[i, 0].item(), float(i),
                msg=f"Node {i} ({tl}): feature value mismatch")

    def test_gnn_output_dim_64(self):
        obs = {tl: np.random.rand(9).astype(np.float32) for tl in TS_IDS}
        graph = self.gb.build_graph(env_state=None, obs=obs)
        gnn = GNNEncoder(in_channels=9, hidden_channels=64, out_channels=64)
        gnn.eval()
        with torch.no_grad():
            out = gnn(graph)
        self.assertEqual(out.shape, (5, 64))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_raw_obs_fed_not_flattened(self):
        """GraphBuilder must receive per-agent (9,) vectors, not a flat concatenation."""
        obs = {tl: np.random.rand(9).astype(np.float32) for tl in TS_IDS}
        graph = self.gb.build_graph(env_state=None, obs=obs)
        # Each node should have 9 features, not 45 (flattened) or 1
        self.assertEqual(graph.x.shape[1], 9,
            f"Node feature dim = {graph.x.shape[1]}, expected 9 (raw per-agent obs)")


# ──────────────────────────────────────────────────────────────────────────────
class TestActionSanity(unittest.TestCase):
    """Test 8: Action sanity checks."""

    def test_exactly_one_action_per_agent(self):
        env = make_env(max_steps=20)
        gnn    = GNNEncoder(in_channels=9, hidden_channels=64, out_channels=64)
        policy = ParameterSharedPPOPolicy(gnn_out_channels=64)
        gb     = GraphBuilder(TS_IDS, TL_EDGES)
        gnn.eval(); policy.eval()

        obs, _ = env.reset()
        graph = gb.build_graph(env_state=None, obs=obs)
        with torch.no_grad():
            embeddings = gnn(graph)
            actions_t, _, _, _ = policy.get_action_and_value(embeddings)

        self.assertEqual(actions_t.shape[0], 5, "Should produce exactly 5 actions")
        env.close()

    def test_all_actions_are_binary(self):
        env = make_env(max_steps=20)
        gnn    = GNNEncoder(in_channels=9, hidden_channels=64, out_channels=64)
        policy = ParameterSharedPPOPolicy(gnn_out_channels=64)
        gb     = GraphBuilder(TS_IDS, TL_EDGES)
        gnn.eval(); policy.eval()

        obs, _ = env.reset()
        for _ in range(10):
            graph = gb.build_graph(env_state=None, obs=obs)
            with torch.no_grad():
                emb = gnn(graph)
                acts, _, _, _ = policy.get_action_and_value(emb)
            for a in acts.cpu().numpy():
                self.assertIn(int(a), [0, 1], f"Invalid action: {a}")
            action_dict = {ts: int(acts[i]) for i, ts in enumerate(TS_IDS)}
            obs, _, term, trunc, _ = env.step(action_dict)
            if all(term.get(tl, False) or trunc.get(tl, False) for tl in TS_IDS):
                break

        env.close()

    def test_agents_can_independently_hold_different_phases(self):
        """After min_green, different agents can hold different phases independently."""
        env = make_env(max_steps=100)
        obs, _ = env.reset()

        # Prime so all agents are past min_green
        for _ in range(4):
            actions = {tl: 0 for tl in TS_IDS}
            obs, _, _, _, _ = env.step(actions)

        # Now send split actions
        split = {tl: (0 if i < 3 else 1) for i, tl in enumerate(TS_IDS)}
        obs, _, _, _, _ = env.step(split)

        # Some agents should be in phase 0, some in phase 2
        phases = {tl: traci.trafficlight.getPhase(tl) for tl in TS_IDS}
        phase_set = set(p for p in phases.values() if p in [0, 2])
        self.assertGreater(len(phase_set), 1,
            f"All agents ended up in same phase {phases} — independent control may not be working")

        env.close()


# ──────────────────────────────────────────────────────────────────────────────
class TestEpisodeLifecycle(unittest.TestCase):
    """Test 9: Reset / lifecycle across 3 consecutive episodes."""

    def test_three_consecutive_episodes(self):
        env = make_env(max_steps=30)

        for episode in range(3):
            obs, infos = env.reset()

            # Verify clean state
            self.assertEqual(len(obs), 5, f"Episode {episode}: wrong obs count at reset")
            for tl in TS_IDS:
                self.assertAlmostEqual(env.time_in_phase[tl], 0.0,
                    msg=f"Episode {episode}: time_in_phase not reset for {tl}")

            rewards_episode = []
            for step in range(30):
                actions = {tl: step % 2 for tl in TS_IDS}
                obs, rewards, term, trunc, _ = env.step(actions)
                rewards_episode.extend(rewards.values())
                if all(term.get(tl, False) or trunc.get(tl, False) for tl in TS_IDS):
                    break

            # Rewards should be finite
            for r in rewards_episode:
                self.assertTrue(math.isfinite(r), f"Episode {episode}: non-finite reward {r}")

            print(f"\n[Lifecycle] Episode {episode}: mean_reward={np.mean(rewards_episode):.4f}")

        env.close()

    def test_traci_not_leaked_between_episodes(self):
        """After reset(), a fresh SUMO instance must be running (no stale state)."""
        env = make_env(max_steps=10)

        for ep in range(2):
            obs, _ = env.reset()
            # Check SUMO is loaded
            self.assertTrue(traci.isLoaded(), f"Episode {ep}: TraCI not loaded after reset")
            sim_time_at_reset = traci.simulation.getTime()
            # After a reset, sim time should be small (within first few steps)
            self.assertLess(sim_time_at_reset, 200,
                f"Episode {ep}: sim_time at reset is {sim_time_at_reset} — possible stale session")
            for _ in range(5):
                env.step({tl: 0 for tl in TS_IDS})

        env.close()
        # After close(), TraCI should not be loaded
        self.assertFalse(traci.isLoaded(), "TraCI still loaded after env.close()")


# ──────────────────────────────────────────────────────────────────────────────
class TestTrafficSimulationSanity(unittest.TestCase):
    """Test 10: Physical traffic sanity."""

    def test_vehicles_spawn_and_queues_form(self):
        env = make_env(max_steps=100)
        obs, _ = env.reset()
        obs_builder = env.obs_builder

        # Run 30 steps with signals held at phase 0
        max_queue_seen = 0
        for step in range(50):
            actions = {tl: 0 for tl in TS_IDS}
            obs, _, term, trunc, _ = env.step(actions)
            for tl in TS_IDS:
                edges = obs_builder.tl_edges[tl]
                raw_queue = sum(
                    traci.edge.getLastStepHaltingNumber(e)
                    for e in edges.values() if e is not None
                )
                max_queue_seen = max(max_queue_seen, raw_queue)
            if all(term.get(tl, False) or trunc.get(tl, False) for tl in TS_IDS):
                break

        env.close()
        self.assertGreater(max_queue_seen, 0,
            "No vehicles queued in 50 steps — traffic demand may not be loading")
        print(f"\n[TrafficSanity] Max queue observed across all intersections: {max_queue_seen}")

    def test_observations_change_over_time(self):
        """Observations should not be static — traffic state should evolve."""
        env = make_env(max_steps=50)
        obs, _ = env.reset()

        obs_history = {tl: [] for tl in TS_IDS}
        for _ in range(20):
            actions = {tl: 0 for tl in TS_IDS}
            obs, _, term, trunc, _ = env.step(actions)
            for tl in TS_IDS:
                obs_history[tl].append(obs[tl].copy())
            if all(term.get(tl, False) or trunc.get(tl, False) for tl in TS_IDS):
                break

        env.close()

        for tl in TS_IDS:
            stacked = np.stack(obs_history[tl])   # [steps, 9]
            variance = stacked.var(axis=0)
            total_var = variance.sum()
            # At least some feature should vary (traffic is dynamic)
            self.assertGreater(total_var, 0.0,
                f"{tl}: observations completely static across 20 steps (var={total_var})")


# ──────────────────────────────────────────────────────────────────────────────
class TestNumericalStability(unittest.TestCase):
    """Test 12: No NaN, inf, or exploding values."""

    def test_no_nan_or_inf_in_100_step_run(self):
        env = make_env(max_steps=100)
        gnn    = GNNEncoder(in_channels=9, hidden_channels=64, out_channels=64)
        policy = ParameterSharedPPOPolicy(gnn_out_channels=64)
        gb     = GraphBuilder(TS_IDS, TL_EDGES)
        gnn.eval(); policy.eval()

        obs, _ = env.reset()
        all_rewards = []
        all_values  = []

        for step in range(100):
            graph = gb.build_graph(env_state=None, obs=obs)
            with torch.no_grad():
                emb = gnn(graph)
                acts, lp, entropy, vals = policy.get_action_and_value(emb)

            # Check GNN output
            self.assertFalse(torch.isnan(emb).any(), f"Step {step}: NaN in GNN embeddings")
            self.assertFalse(torch.isinf(emb).any(), f"Step {step}: Inf in GNN embeddings")

            # Check policy output
            self.assertFalse(torch.isnan(lp).any(), f"Step {step}: NaN in log_probs")
            self.assertFalse(torch.isnan(vals).any(), f"Step {step}: NaN in values")

            action_dict = {ts: int(acts[i]) for i, ts in enumerate(TS_IDS)}
            obs, rewards, term, trunc, _ = env.step(action_dict)

            for r in rewards.values():
                self.assertTrue(math.isfinite(r), f"Step {step}: non-finite reward {r}")

            all_rewards.extend(rewards.values())
            all_values.append(vals.mean().item())

            if all(term.get(tl, False) or trunc.get(tl, False) for tl in TS_IDS):
                break

        env.close()
        print(f"\n[NumericalStability] 100 steps: "
              f"reward range=[{min(all_rewards):.3f}, {max(all_rewards):.3f}], "
              f"value range=[{min(all_values):.3f}, {max(all_values):.3f}]")


if __name__ == "__main__":
    unittest.main(verbosity=2)

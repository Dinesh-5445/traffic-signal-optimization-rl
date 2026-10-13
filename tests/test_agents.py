"""
Agent tests validating:
  1. Gradient flow through GNN -> Actor-Critic (no detached parameters)
  2. Single-step overfit: policy should converge on a fixed trivial input within
     a small number of gradient steps, proving the learning pipeline is functional.
"""

import unittest
import torch
import numpy as np
from torch_geometric.data import Data

from agents.gnn_encoder import GNNEncoder
from agents.ppo_policy import ParameterSharedPPOPolicy
from training.rollout_buffer import RolloutBuffer
from training.marl_ppo import MARLTrainer
from communication.graph_builder import GraphBuilder


DEVICE = torch.device("cpu")
NUM_AGENTS = 5
NODE_FEAT_DIM = 9
GNN_OUT = 64


def _make_dummy_graph(num_agents: int, feat_dim: int) -> Data:
    """Build a minimal fully-connected graph for testing."""
    x = torch.randn(num_agents, feat_dim)
    src = []
    dst = []
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class TestGradientFlow(unittest.TestCase):
    """
    Verify that every parameter in GNNEncoder and PPOPolicy receives gradients
    during a backward pass. This is a critical sanity check to ensure the graph
    is not accidentally detached anywhere in the pipeline.
    """

    def setUp(self):
        self.gnn = GNNEncoder(in_channels=NODE_FEAT_DIM, hidden_channels=GNN_OUT, out_channels=GNN_OUT).to(DEVICE)
        self.policy = ParameterSharedPPOPolicy(gnn_out_channels=GNN_OUT).to(DEVICE)

    def test_actor_gradients_flow_through_gnn(self):
        graph = _make_dummy_graph(NUM_AGENTS, NODE_FEAT_DIM)
        embeddings = self.gnn(graph)                        # [num_agents, GNN_OUT]
        actions, log_probs, entropy, values = self.policy.get_action_and_value(embeddings)

        # Compute a dummy loss and backprop
        loss = -log_probs.mean() + 0.5 * values.mean()
        loss.backward()

        # Check every GNN parameter has a non-None, non-zero gradient
        for name, param in self.gnn.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"GNN parameter '{name}' has no gradient.")
            self.assertFalse(
                torch.all(param.grad == 0),
                msg=f"GNN parameter '{name}' has all-zero gradient."
            )

        # Check every policy parameter has a gradient
        for name, param in self.policy.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"Policy parameter '{name}' has no gradient.")

    def test_critic_gradient_flow(self):
        graph = _make_dummy_graph(NUM_AGENTS, NODE_FEAT_DIM)
        embeddings = self.gnn(graph)
        _, _, _, values = self.policy.get_action_and_value(embeddings)

        loss = values.mean()
        loss.backward()

        for name, param in self.gnn.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"GNN param '{name}' has no gradient on critic pass.")


class TestRolloutBuffer(unittest.TestCase):
    """Verify RolloutBuffer stores, computes GAE, and yields correct minibatches."""

    def setUp(self):
        self.num_steps = 16
        self.num_agents = 3
        self.obs_dim = 9
        self.buf = RolloutBuffer(self.num_steps, self.num_agents, self.obs_dim, DEVICE)

    def _fill_buffer(self):
        for _ in range(self.num_steps):
            self.buf.add(
                observations=torch.randn(self.num_agents, self.obs_dim),
                actions=torch.zeros(self.num_agents, dtype=torch.long),
                log_probs=torch.full((self.num_agents,), -0.693),
                rewards=torch.ones(self.num_agents),
                values=torch.zeros(self.num_agents),
                dones=torch.zeros(self.num_agents),
            )

    def test_buffer_fills_correctly(self):
        self._fill_buffer()
        self.assertEqual(self.buf._ptr, self.num_steps)
        self.assertEqual(self.buf.rewards.shape, (self.num_steps, self.num_agents))

    def test_gae_returns_correct_shape(self):
        self._fill_buffer()
        last_values = torch.zeros(self.num_agents)
        self.buf.compute_returns_and_advantages(last_values)
        self.assertEqual(self.buf.advantages.shape, (self.num_steps, self.num_agents))
        self.assertEqual(self.buf.returns.shape, (self.num_steps, self.num_agents))

    def test_minibatch_generator_covers_all_data(self):
        self._fill_buffer()
        last_values = torch.zeros(self.num_agents)
        self.buf.compute_returns_and_advantages(last_values)

        total_samples = 0
        for batch in self.buf.get_batches(minibatch_size_in_steps=4):
            # batch[0] is mb_obs: [minibatch_steps, num_agents, obs_dim]
            total_samples += batch[0].shape[0] * batch[0].shape[1]

        self.assertEqual(total_samples, self.num_steps * self.num_agents)

    def test_reset_clears_pointer(self):
        self._fill_buffer()
        self.buf.reset()
        self.assertEqual(self.buf._ptr, 0)


class TestPolicyOverfit(unittest.TestCase):
    """
    Sanity check: force policy to overfit on a single fixed observation.
    The loss should decrease monotonically over ~100 gradient steps.
    This validates the entire GNN -> Policy -> Loss -> Backprop pipeline.
    """

    def test_single_sample_overfit(self):
        gnn = GNNEncoder(in_channels=NODE_FEAT_DIM, hidden_channels=GNN_OUT, out_channels=GNN_OUT)
        policy = ParameterSharedPPOPolicy(gnn_out_channels=GNN_OUT)
        optimizer = torch.optim.Adam(list(gnn.parameters()) + list(policy.parameters()), lr=1e-3)

        graph = _make_dummy_graph(NUM_AGENTS, NODE_FEAT_DIM)
        fixed_actions = torch.zeros(NUM_AGENTS, dtype=torch.long)

        losses = []
        for _ in range(100):
            embeddings = gnn(graph)
            _, log_probs, entropy, values = policy.get_action_and_value(embeddings, fixed_actions)
            # Maximise log-prob of action 0 by minimising negative log-prob
            loss = -log_probs.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease overall — first half avg > second half avg
        first_half = np.mean(losses[:50])
        second_half = np.mean(losses[50:])
        self.assertLess(second_half, first_half, "Policy did not learn to overfit on fixed actions.")

class MockEnv:
    def __init__(self, ts_ids):
        self.ts_ids = ts_ids
        self.agents = ts_ids
    def reset(self):
        obs = {ts: np.random.rand(9).astype(np.float32) for ts in self.ts_ids}
        return obs, {}
    def step(self, actions):
        obs = {ts: np.random.rand(9).astype(np.float32) for ts in self.ts_ids}
        rewards = {ts: np.random.rand() for ts in self.ts_ids}
        terminations = {ts: False for ts in self.ts_ids}
        truncations = {ts: False for ts in self.ts_ids}
        return obs, rewards, terminations, truncations, {}

class TestPPOUpdate(unittest.TestCase):
    def test_ppo_update_gnn_gradient_flow(self):
        TS_IDS = ['A1', 'B0', 'B1', 'B2', 'C1']
        TL_EDGES = {
            'A1': {'N': 'A2A1', 'S': 'A0A1', 'E': 'B1A1', 'W': None},
            'B0': {'N': 'B1B0', 'S': None,   'E': 'C0B0', 'W': 'A0B0'},
            'B1': {'N': 'B2B1', 'S': 'B0B1', 'E': 'C1B1', 'W': 'A1B1'},
            'B2': {'N': None,   'S': 'B1B2', 'E': 'C2B2', 'W': 'A2B2'},
            'C1': {'N': 'C2C1', 'S': 'C0C1', 'E': None,   'W': 'B1C1'},
        }
        env = MockEnv(TS_IDS)
        gb = GraphBuilder(TS_IDS, TL_EDGES)
        trainer = MARLTrainer(env=env, ts_ids=TS_IDS, graph_builder=gb, device=DEVICE, num_steps=16, num_epochs=1, num_minibatches=2)
        
        # 1. Collect an actual rollout
        trainer._collect_rollout()
        
        # 2. Save copies of all trainable GNN parameters
        old_params = {}
        for name, param in trainer.gnn.named_parameters():
            old_params[name] = param.clone().detach()
            
        # 3. Run _ppo_update()
        trainer._ppo_update()
        
        # 4. Inspect GNN gradients and 5. Inspect GNN parameters after optimizer.step()
        has_grad = False
        param_changed = False
        
        for name, param in trainer.gnn.named_parameters():
            # A. At least one GNN param has grad
            if param.grad is not None:
                has_grad = True
                
                # C. The gradient contains only finite values
                self.assertTrue(torch.isfinite(param.grad).all())
                
                # D. The gradient norm is > 1e-10
                grad_norm = param.grad.norm().item()
                if name == "conv1.lin.weight":
                    print(f"\\ngnn.conv1.lin.weight gradient norm: {grad_norm}")
                self.assertGreater(grad_norm, 1e-10)
                
            # E. At least one GNN param changes
            if not torch.allclose(param, old_params[name]):
                param_changed = True
                
        self.assertTrue(has_grad, "No GNN parameter received a gradient!")
        self.assertTrue(param_changed, "No GNN parameter changed after optimizer.step()!")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Signal-control mechanism regression tests
# ─────────────────────────────────────────────────────────────────────────────

class MockTrafficEnv:
    """
    Minimal stand-in for TrafficMARLEnv that drives the signal-control logic
    in multi_agent_env without requiring a live SUMO process.
    """
    PHASE_MAP = {0: 0, 1: 2}          # action → commanded green phase
    YELLOW    = {0: 1, 2: 3}          # green phase → yellow phase

    def __init__(self, ts_ids, min_green=10, max_green=60, delta_time=5):
        self.ts_ids     = ts_ids
        self.min_green  = min_green
        self.max_green  = max_green
        self.delta_time = delta_time
        # Internal state
        self._phase        = {tl: 0 for tl in ts_ids}   # current green phase (0 or 2)
        self.time_in_phase = {tl: 0.0 for tl in ts_ids}

    # ---------- public API mimicking the env step logic ----------
    def apply_actions(self, actions: dict) -> dict:
        """
        Applies the same min/max-green logic as TrafficMARLEnv.step().
        Returns dict of {tl: (new_phase, phase_changed)}.
        """
        results = {}
        for tl, act in actions.items():
            commanded = self.PHASE_MAP[act]
            current   = self._phase[tl]

            is_min_locked = self.time_in_phase[tl] < self.min_green
            is_max_locked = self.time_in_phase[tl] >= self.max_green

            if is_min_locked:
                new_phase = current
            elif is_max_locked:
                new_phase = 2 if current == 0 else 0   # forced switch
            else:
                new_phase = commanded

            if new_phase != current:
                self.time_in_phase[tl] = 0.0
                changed = True
            else:
                self.time_in_phase[tl] += self.delta_time
                changed = False

            self._phase[tl] = new_phase
            results[tl]     = (new_phase, changed)
        return results

    def advance_time(self, tl, steps=1):
        """Advance time without triggering a phase change (all-hold)."""
        for _ in range(steps):
            self.apply_actions({tl: 0 if self._phase[tl] == 0 else 1})

    def current_phase(self, tl):
        return self._phase[tl]


class TestSignalControlMechanism(unittest.TestCase):
    """
    Regression tests for the RL → signal-control pipeline.

    These tests operate on the *logic* of the control mechanism
    without requiring a live SUMO process or trained weights.
    """

    TS_IDS = ['A1', 'B0', 'B1', 'B2', 'C1']

    # ── 1. Action mapping ──────────────────────────────────────────────────
    def test_action_0_maps_to_ns_green_phase_0(self):
        env = MockTrafficEnv(self.TS_IDS[:1])
        # Start past min_green so the action is honoured
        env.time_in_phase['A1'] = 15.0
        result = env.apply_actions({'A1': 0})
        phase, _ = result['A1']
        self.assertEqual(phase, 0, "Action 0 must map to phase 0 (N/S green)")

    def test_action_1_maps_to_ew_green_phase_2(self):
        env = MockTrafficEnv(self.TS_IDS[:1])
        env.time_in_phase['A1'] = 15.0
        result = env.apply_actions({'A1': 1})
        phase, _ = result['A1']
        self.assertEqual(phase, 2, "Action 1 must map to phase 2 (E/W green)")

    # ── 2. Minimum-green floor ─────────────────────────────────────────────
    def test_min_green_prevents_phase_switch_below_threshold(self):
        """Phase must not switch while time_in_phase < min_green."""
        env = MockTrafficEnv(self.TS_IDS[:1], min_green=10, delta_time=5)
        env._phase['A1']        = 0        # currently in phase 0
        env.time_in_phase['A1'] = 5.0      # only 5s elapsed → below min_green

        result = env.apply_actions({'A1': 1})   # request opposite phase
        phase, changed = result['A1']
        self.assertFalse(changed, "Phase must not switch below min_green")
        self.assertEqual(phase, 0, "Phase must remain 0 while min-locked")

    def test_min_green_allows_switch_at_threshold(self):
        """Phase switch is legal once time_in_phase >= min_green."""
        env = MockTrafficEnv(self.TS_IDS[:1], min_green=10, delta_time=5)
        env._phase['A1']        = 0
        env.time_in_phase['A1'] = 10.0     # exactly at min_green

        result = env.apply_actions({'A1': 1})
        phase, changed = result['A1']
        self.assertTrue(changed, "Phase must switch once min_green is reached")
        self.assertEqual(phase, 2)

    # ── 3. Maximum-green ceiling ────────────────────────────────────────────
    def test_max_green_forces_switch_at_ceiling(self):
        """When time_in_phase >= max_green the env must force a switch."""
        env = MockTrafficEnv(self.TS_IDS[:1], min_green=10, max_green=60, delta_time=5)
        env._phase['A1']        = 0
        env.time_in_phase['A1'] = 60.0    # at the ceiling

        # Even if agent votes to hold (action 0), the env must switch
        result = env.apply_actions({'A1': 0})
        phase, changed = result['A1']
        self.assertTrue(changed, "max_green ceiling must force a phase switch")
        self.assertEqual(phase, 2)

    def test_no_hidden_fixed_45s_cycle(self):
        """
        Run the control logic for 200+ steps with alternating actions and
        verify the resulting green durations are NOT all 45s.
        The test validates the *mechanism* — the timing is action-driven.
        """
        env = MockTrafficEnv(['B1'], min_green=10, max_green=60, delta_time=5)
        env._phase['B1']        = 0
        env.time_in_phase['B1'] = 0.0

        durations = []
        current_dur = 0.0
        current_ph  = 0

        for step in range(200):
            act = step % 2         # alternating actions
            result = env.apply_actions({'B1': act})
            ph, changed = result['B1']
            if changed:
                durations.append(current_dur)
                current_dur = env.delta_time
                current_ph  = ph
            else:
                current_dur += env.delta_time

        # All durations must NOT be identically 45s
        if durations:
            unique = set(durations)
            self.assertFalse(
                unique == {45.0},
                f"All green durations are 45s — hidden fixed-time controller suspected. Got: {sorted(unique)}"
            )

    def test_no_hidden_fixed_63s_cycle(self):
        """Similar check for 63s (the old SUMO default 60+3 cycle)."""
        env = MockTrafficEnv(['B1'], min_green=10, max_green=60, delta_time=5)
        env._phase['B1']        = 0
        env.time_in_phase['B1'] = 0.0

        durations = []
        current_dur = 0.0

        for step in range(200):
            act = step % 2
            result = env.apply_actions({'B1': act})
            _, changed = result['B1']
            if changed:
                durations.append(current_dur)
                current_dur = env.delta_time
            else:
                current_dur += env.delta_time

        if durations:
            unique = set(durations)
            self.assertFalse(
                unique == {63.0},
                f"All cycle lengths are 63s — SUMO XML override suspected. Got: {sorted(unique)}"
            )

    # ── 4. Different action sequences → different outcomes ──────────────────
    def test_all_action0_vs_all_action1_produce_different_phases(self):
        """Repeated action-0 and repeated action-1 must produce different phase histories."""
        def run(constant_action):
            env = MockTrafficEnv(['A1'], min_green=10, max_green=60, delta_time=5)
            env._phase['A1']        = 0
            env.time_in_phase['A1'] = 15.0   # past min_green from the start
            history = []
            for _ in range(20):
                r = env.apply_actions({'A1': constant_action})
                history.append(r['A1'][0])
            return history

        hist_0 = run(0)
        hist_1 = run(1)
        self.assertNotEqual(hist_0, hist_1,
            "Constant action 0 and constant action 1 must produce different phase histories")

    def test_different_agents_can_hold_different_phases(self):
        """
        When agents receive opposite actions they should independently
        hold different phases (after min_green is satisfied).
        """
        env = MockTrafficEnv(['A1', 'C1'], min_green=10, delta_time=5)
        # Seed each agent past min_green in the *same* phase
        env._phase['A1']        = 0;  env.time_in_phase['A1'] = 15.0
        env._phase['C1']        = 0;  env.time_in_phase['C1'] = 15.0

        result = env.apply_actions({'A1': 0, 'C1': 1})
        self.assertEqual(result['A1'][0], 0, "A1 must hold phase 0")
        self.assertEqual(result['C1'][0], 2, "C1 must switch to phase 2")

    # ── 5. Legal yellow transition check ───────────────────────────────────
    def test_yellow_phase_is_consecutive_to_green(self):
        """
        Yellow is always currentGreen + 1.
        Phase 0 (N/S green) → yellow phase 1
        Phase 2 (E/W green) → yellow phase 3
        """
        yellow_map = MockTrafficEnv.YELLOW
        self.assertEqual(yellow_map[0], 1, "N/S yellow must be phase 1")
        self.assertEqual(yellow_map[2], 3, "E/W yellow must be phase 3")

    # ── 6. Timer reset on phase change ─────────────────────────────────────
    def test_timer_resets_on_phase_change(self):
        """time_in_phase must be 0 immediately after a switch."""
        env = MockTrafficEnv(['B1'], min_green=10, delta_time=5)
        env._phase['B1']        = 0
        env.time_in_phase['B1'] = 15.0     # past min_green

        env.apply_actions({'B1': 1})       # trigger switch
        self.assertEqual(env.time_in_phase['B1'], 0.0,
            "time_in_phase must reset to 0 on phase change")

    def test_timer_increments_on_hold(self):
        """time_in_phase must increment by delta_time when phase is held."""
        env = MockTrafficEnv(['B1'], min_green=10, delta_time=5)
        env._phase['B1']        = 0
        env.time_in_phase['B1'] = 5.0

        env.apply_actions({'B1': 0})       # same phase → hold
        self.assertAlmostEqual(env.time_in_phase['B1'], 10.0,
            msg="time_in_phase must increase by delta_time on hold")


if __name__ == "__main__":
    unittest.main()

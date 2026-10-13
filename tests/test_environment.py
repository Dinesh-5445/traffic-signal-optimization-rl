import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from environment.builders import ObsBuilder, MultiObjectiveRewardBuilder
from communication.graph_builder import GraphBuilder
from environment.multi_agent_env import TrafficMARLEnv

from pettingzoo.test import parallel_api_test

class TestBuilders(unittest.TestCase):
    def setUp(self):
        self.ts_ids = ['A1', 'B1']
        self.tl_edges = {
            'A1': {'N': 'N_A1', 'S': 'S_A1', 'E': 'E_A1', 'W': 'W_A1'},
            'B1': {'N': 'N_B1', 'S': 'S_B1', 'E': 'E_B1', 'W': 'W_B1'}
        }
        self.neighbor_map = {'A1': ['B1'], 'B1': ['A1']}
        
    @patch('environment.builders.traci')
    def test_obs_builder_raw(self, mock_traci):
        mock_traci.edge.getLastStepHaltingNumber.return_value = 5.0
        mock_traci.edge.getWaitingTime.return_value = 10.0
        mock_traci.trafficlight.getPhase.return_value = 2
        
        # Testing with normalize=False (backward compatibility)
        builder = ObsBuilder(self.ts_ids, self.tl_edges, normalize=False)
        obs = builder.build_observation()
        
        self.assertIn('A1', obs)
        self.assertIn('B1', obs)
        self.assertEqual(len(obs['A1']), 9)
        self.assertTrue(np.allclose(obs['A1'], [5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 2.0]))
        
    @patch('environment.builders.traci')
    def test_obs_builder_normalized(self, mock_traci):
        mock_traci.edge.getLastStepHaltingNumber.return_value = 5.0
        mock_traci.edge.getWaitingTime.return_value = 10.0
        mock_traci.trafficlight.getPhase.return_value = 2
        
        # Testing default normalize=True
        builder = ObsBuilder(self.ts_ids, self.tl_edges)
        obs = builder.build_observation()
        
        expected = [
            5.0 / 20.0, 5.0 / 20.0, 5.0 / 20.0, 5.0 / 20.0,
            10.0 / 500.0, 10.0 / 500.0, 10.0 / 500.0, 10.0 / 500.0,
            2.0 / 3.0
        ]
        self.assertIn('A1', obs)
        self.assertTrue(np.allclose(obs['A1'], expected))
        
    @patch('environment.builders.traci')
    def test_reward_builder_unscaled(self, mock_traci):
        mock_traci.edge.getLastStepHaltingNumber.return_value = 2.0
        mock_traci.edge.getWaitingTime.return_value = 5.0
        mock_traci.edge.getLastStepVehicleNumber.return_value = 10
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0
        mock_traci.trafficlight.getPhase.return_value = 0
        
        # Testing with scales = 1.0 (backward compatibility)
        builder = MultiObjectiveRewardBuilder(
            self.ts_ids, self.tl_edges, self.neighbor_map,
            reward_scale=1.0, wait_scale=1.0, queue_scale=1.0
        )
        builder.reset()
        rewards = builder.compute_reward()
        
        # queue penalty = -1.0 * 8 = -8.0
        # wait penalty = -0.5 * 20 = -10.0
        # throughput = 0
        # switch = 0
        # teleport = 0
        # local reward = -18.0
        # B1 neighbor: -18.0. average neighbor reward = -18.0.
        # coop reward = -18.0 + 0.2 * (-18.0) = -21.6
        self.assertAlmostEqual(rewards['A1'], -21.6)

    @patch('environment.builders.traci')
    def test_reward_builder_scaled(self, mock_traci):
        mock_traci.edge.getLastStepHaltingNumber.return_value = 2.0
        mock_traci.edge.getWaitingTime.return_value = 5.0
        mock_traci.edge.getLastStepVehicleNumber.return_value = 10
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0
        mock_traci.trafficlight.getPhase.return_value = 0
        
        # Testing default scaling
        builder = MultiObjectiveRewardBuilder(self.ts_ids, self.tl_edges, self.neighbor_map)
        builder.reset()
        rewards = builder.compute_reward()
        
        # queue penalty = -1.0 * (8 * 0.1) = -0.8
        # wait penalty = -0.5 * (20 * 0.01) = -0.1
        # throughput = 0, switch = 0, teleport = 0
        # local reward = -0.9 * 0.1 = -0.09
        # B1 neighbor: -0.09. average neighbor reward = -0.09.
        # coop reward = -0.09 + 0.2 * (-0.09) = -0.108
        self.assertAlmostEqual(rewards['A1'], -0.108)
        
    def test_graph_builder(self):
        builder = GraphBuilder(self.ts_ids, self.tl_edges)
        obs = {'A1': np.zeros(9), 'B1': np.ones(9)}
        data = builder.build_graph(None, obs)
        
        self.assertEqual(data.x.shape, (2, 9))
        self.assertEqual(data.edge_index.shape[0], 2)

@patch('environment.builders.traci')
@patch('environment.multi_agent_env.traci')
def test_pettingzoo_api(mock_env_traci, mock_builders_traci):
    # Mock SUMO interactions to pass the PettingZoo validation
    mock_env_traci.isLoaded.return_value = False
    mock_env_traci.simulation.getMinExpectedNumber.return_value = 10
    
    mock_builders_traci.simulation.getStartingTeleportNumber.return_value = 0
    mock_builders_traci.edge.getLastStepHaltingNumber.return_value = 0.0
    mock_builders_traci.edge.getWaitingTime.return_value = 0.0
    mock_builders_traci.edge.getLastStepVehicleNumber.return_value = 0
    mock_builders_traci.trafficlight.getPhase.return_value = 0
    
    mock_env_traci.trafficlight.getPhase.return_value = 0
    mock_env_traci.simulation.getTime.return_value = 0.0
    
    ts_ids = ['A1', 'B1']
    tl_edges = {
        'A1': {'N': 'N_A1', 'S': 'S_A1', 'E': 'E_A1', 'W': 'W_A1'},
        'B1': {'N': 'N_B1', 'S': 'S_B1', 'E': 'E_B1', 'W': 'W_B1'}
    }
    neighbor_map = {'A1': ['B1'], 'B1': ['A1']}
    
    obs_b = ObsBuilder(ts_ids, tl_edges)
    rew_b = MultiObjectiveRewardBuilder(ts_ids, tl_edges, neighbor_map)
    graph_b = GraphBuilder(ts_ids, tl_edges)
    
    env = TrafficMARLEnv(
        sumo_cmd=["sumo"],
        ts_ids=ts_ids,
        obs_builder=obs_b,
        reward_builder=rew_b,
        graph_builder=graph_b
    )
    
    parallel_api_test(env, num_cycles=10)

if __name__ == '__main__':
    unittest.main()

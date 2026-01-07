import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAgentImport(unittest.TestCase):
    
    @patch('agent.agent.EmulatorHarness')
    @patch('agent.agent.CFG')
    def test_planning_agent_initialization(self, mock_cfg, mock_emulator):
        """
        Test that PlanningAgent can be initialized.
        We mock the dependencies to avoid needing a ROM or actual emulator.
        """
        # Setup mocks
        mock_llm = MagicMock()
        mock_cfg.ADAPTIVE_COMPUTE_ENABLED = False
        mock_cfg.LLM_PLANNING_BUDGET_DEFAULT = 100
        
        # Import inside test to ensure mocks are active
        from agent.agent import PlanningAgent
        
        agent = PlanningAgent(llm_client=mock_llm)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.llm, mock_llm)

if __name__ == '__main__':
    unittest.main()

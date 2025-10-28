"""
Agent 2 wrapper for battle system - loads the same trained RL model as Agent 1
"""

import sys
import os
sys.path.insert(0, '/Users/sujoydas/UTMIST-AI2')

from user.train_agent import SB3Agent
from stable_baselines3 import PPO

class SubmittedAgent(SB3Agent):
    """Agent wrapper that loads the same trained model as Agent 1"""

    def __init__(self):
        # Load the same trained model as Agent 1
        model_path = '/Users/sujoydas/UTMIST-AI2/checkpoints/experiment_aggressive/rl_model_41909028_steps'
        super().__init__(sb3_class=PPO, file_path=model_path)

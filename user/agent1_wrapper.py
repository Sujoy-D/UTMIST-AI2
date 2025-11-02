"""
Agent 1 wrapper for battle system - loads trained RL model
"""

import sys
import os
sys.path.insert(0, '/Users/sujoydas/UTMIST-AI2')

from user.train_agent import SB3Agent
from stable_baselines3 import PPO

class SubmittedAgent(SB3Agent):
    """Agent wrapper that loads a specific trained model"""

    def __init__(self):
        # Load your trained model here - replace with your actual model path
        model_path = '/Users/sujoydas/UTMIST-AI2/checkpoints/experiment_aggressive/rl_model_41909028_steps'
        super().__init__(sb3_class=PPO, file_path=model_path)

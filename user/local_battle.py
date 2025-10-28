#!/usr/bin/env python3
"""
Simple local battle script that doesn't require database setup
"""

import os
import sys
from loguru import logger
import importlib.util

# Add the project root to the path
sys.path.insert(0, '/Users/sujoydas/UTMIST-AI2')

from environment.agent import run_match, CameraResolution
from user.train_agent import gen_reward_manager

def load_agent_class(file_path):
    """Dynamically load SubmittedAgent class from a given Python file."""
    file_path = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load module spec and import dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Expecting a class named SubmittedAgent in the file
    if not hasattr(module, "SubmittedAgent"):
        raise AttributeError(f"File {file_path} does not define a SubmittedAgent class.")

    return module.SubmittedAgent

def run_local_battle(agent1_path, agent2_path):
    """Run a battle between two agents locally without database tracking"""
    
    print(f"🥊 Starting Local Battle")
    print(f"Agent 1: {agent1_path}")
    print(f"Agent 2: {agent2_path}")
    print("-" * 50)
    
    try:
        # Load agent classes
        logger.info("Loading agents...")
        Agent1 = load_agent_class(agent1_path)
        Agent2 = load_agent_class(agent2_path)

        # Instantiate agents
        logger.info("Creating agent instances...")
        agent1_instance = Agent1()
        agent2_instance = Agent2()
        
        # Create reward manager
        reward_manager = gen_reward_manager()
        
        # Set match parameters
        match_time = 90  # seconds
        
        logger.info("✅ Both agents successfully instantiated.")
        logger.info(f"🚀 Starting {match_time}s match...")
        
        # Run the match
        match_result = run_match(
            agent1_instance,
            agent_2=agent2_instance,
            video_path='local_battle.mp4',
            agent_1_name='Trained RL Agent 1',
            agent_2_name='Trained RL Agent 2',
            resolution=CameraResolution.LOW,
            reward_manager=reward_manager,
            max_timesteps=30 * match_time,  # 30 FPS * seconds
            train_mode=False  # Set to False for battle mode
        )
        
        print("\n" + "="*50)
        print("🏆 BATTLE RESULTS")
        print("="*50)
        print(f"Match Result: {match_result}")
        print(f"Player 1 Result: {match_result.player1_result}")
        print(f"Player 2 Result: {match_result.player2_result}")
        print(f"Video saved as: local_battle.mp4")
        print("="*50)
        
        return match_result
        
    except Exception as e:
        logger.error(f"Battle failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Battle configurations
    agent1_path = "/Users/sujoydas/UTMIST-AI2/user/agent1_wrapper.py"
    agent2_path = "/Users/sujoydas/UTMIST-AI2/user/agent2_wrapper.py"
    
    # Run the battle
    result = run_local_battle(agent1_path, agent2_path)
    
    if result:
        print("✅ Battle completed successfully!")
    else:
        print("❌ Battle failed!")

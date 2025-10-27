# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.weapon_ranges = {"Hammer": 2.0, "Spear": 3.0, "Punch": 1.0}
        self.stage_width = 10.67

    def predict(self, obs):
        self.time += 1

        # Extract observation data
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        player_state = self.obs_helper.get_section(obs, 'player_state')
        opp_state = self.obs_helper.get_section(obs, 'opponent_state')
        player_weapon = self.obs_helper.get_section(obs, 'player_weapon')

        # Calculate key distances
        distance_to_opponent = abs(pos[0] - opp_pos[0])
        stage_edge = self.stage_width / 2
        distance_from_edge = stage_edge - abs(pos[0])

        action = self.act_helper.zeros()
        optimal_range = self.weapon_ranges.get(player_weapon, 2.0)
        danger_zone = 1.5

        # === PRIORITY 1: STAGE SAFETY (Don't fall off!) ===
        if distance_from_edge < 1.0:  # Too close to edge - EMERGENCY!
            if pos[0] > 0:  # On right edge
                action = self.act_helper.press_keys(['a'])  # Move left toward center
            else:  # On left edge
                action = self.act_helper.press_keys(['d'])  # Move right toward center

            # Defensive jump when near edge
            if self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)
            return action  # Return immediately - stage safety is top priority

        # === PRIORITY 2: HEIGHT SAFETY (Don't fall from height) ===
        if pos[1] < -2.0:  # Getting too low
            # Try to jump to recover
            if self.time % 3 == 0:
                action = self.act_helper.press_keys(['space'], action)
            return action

        # === PRIORITY 3: EVASION & COMBAT STRATEGY ===

        # Situation 1: Opponent too close - EVADE!
        if distance_to_opponent < danger_zone:
            # Smart evasion that considers stage edges
            if opp_pos[0] > pos[0]:  # Opponent on right
                # Move left if safe, otherwise move right
                if pos[0] > -stage_edge + 1.5:
                    action = self.act_helper.press_keys(['a'])
                else:
                    action = self.act_helper.press_keys(['d'])
            else:  # Opponent on left
                # Move right if safe, otherwise move left
                if pos[0] < stage_edge - 1.5:
                    action = self.act_helper.press_keys(['d'])
                else:
                    action = self.act_helper.press_keys(['a'])

            # Evasive jump
            if self.time % 3 == 0:
                action = self.act_helper.press_keys(['space'], action)

        # Situation 2: In optimal combat range
        elif distance_to_opponent < optimal_range:
            # Opponent is vulnerable - ATTACK!
            if opp_state in [5, 11]:  # KO'd or stunned
                action = self.act_helper.press_keys(['j'])
            else:
                # Maintain optimal distance while drifting toward center
                if pos[0] > 0:  # On right side - drift left toward center
                    action = self.act_helper.press_keys(['a'])
                else:  # On left side - drift right toward center
                    action = self.act_helper.press_keys(['d'])

        # Situation 3: Too far - close distance cautiously
        else:
            if opp_pos[0] > pos[0]:  # Opponent on right
                if pos[0] < stage_edge - 1.0:  # Check right edge safety
                    action = self.act_helper.press_keys(['d'])
            else:  # Opponent on left
                if pos[0] > -stage_edge + 1.0:  # Check left edge safety
                    action = self.act_helper.press_keys(['a'])

        # === DEFENSIVE REACTION: Jump when opponent attacks near you ===
        if (opp_state == AttackState and distance_to_opponent < 2.5 and
                distance_from_edge < 2.0 and self.time % 2 == 0):
            action = self.act_helper.press_keys(['space'], action)

        return action

    # Keep the existing methods from the original SubmittedAgent
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
        else:
            self.model = PPO.load(self.file_path)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

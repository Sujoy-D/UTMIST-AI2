'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below.

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 256,  # Reduced from 512 for performance
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=2048,  # Reduced from 54,000 to 2,048 for much faster episodes
                                      batch_size=64,   # Increased from 16 to 64 for better efficiency
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if falling too low (Y+ is downward) or opponent is below you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            action_dim=10,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )

class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def unified_positioning_reward(
    env: WarehouseBrawl,
    optimal_range: float = 2.5,
    max_range: float = 5.0,
    approach_weight: float = 3.0,
    attack_weight: float = 5.0,
    focus_radius: float = 4.0,
    combat_bonus: float = 3.0,
    positioning_bonus: float = 1.5
) -> float:
    """
    Unified positioning reward that combines:
    - Aggressive positioning (maintaining optimal attack range)
    - Approach and engagement (moving toward opponent and attacking when close)
    - Combat focus (staying engaged in combat)
    - Opponent tracking (actively seeking opponent when not attacking)
    - Positioning before attack (good setup for attacks)

    Args:
        env (WarehouseBrawl): The game environment.
        optimal_range (float): The ideal distance for attacking.
        max_range (float): Maximum range before heavy penalty.
        approach_weight (float): Reward weight for approaching opponent.
        attack_weight (float): Reward weight for attacking when close.
        focus_radius (float): Radius within which combat is considered focused.
        combat_bonus (float): Bonus for combat actions within focus radius.
        positioning_bonus (float): Bonus for good positioning near opponent.

    Returns:
        float: The computed unified positioning reward.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Calculate distances
    current_distance = ((player.body.position.x - opponent.body.position.x)**2 +
                       (player.body.position.y - opponent.body.position.y)**2)**0.5
    prev_distance = ((player.prev_x - opponent.body.position.x)**2 +
                    (player.prev_y - opponent.body.position.y)**2)**0.5

    reward = 0.0
    player_attacking = isinstance(player.state, AttackState)

    # 1. AGGRESSIVE POSITIONING - reward for being in optimal range
    if current_distance <= optimal_range:
        reward += 1.0
    elif current_distance <= max_range:
        reward += 1.0 - (current_distance - optimal_range) / (max_range - optimal_range)
    else:
        reward -= 0.5

    # 2. APPROACH ENGAGEMENT - reward moving toward opponent
    distance_change = prev_distance - current_distance
    if distance_change > 0:  # Got closer
        approach_reward = min(distance_change * approach_weight, approach_weight)
        reward += approach_reward

    # 3. ATTACK WHEN CLOSE - heavily reward attacking when in optimal range
    if player_attacking and current_distance <= optimal_range:
        reward += attack_weight

    # 4. BONUS FOR BEING IN OPTIMAL RANGE
    if current_distance <= optimal_range:
        reward += 1.0

    # 5. COMBAT FOCUS - reward staying engaged
    if current_distance <= focus_radius:
        # Reward for being in combat range
        reward += positioning_bonus

        # Extra reward for attacking while in range
        if player_attacking:
            reward += combat_bonus

        # Reward for facing opponent
        opponent_direction = 1 if opponent.body.position.x > player.body.position.x else -1
        if (player.facing.value * opponent_direction) > 0:  # Facing toward opponent
            reward += 0.5
    else:
        # Penalize being too far from opponent
        distance_penalty = min((current_distance - focus_radius) * 0.5, 2.0)
        reward -= distance_penalty

    # 6. OPPONENT TRACKING - reward moving closer when not attacking
    if not player_attacking:
        if distance_change > 0:  # Got closer
            reward += min(distance_change, 1.0)
        else:  # Got further or stayed same
            reward += max(distance_change, -0.5)

    # 7. POSITIONING BEFORE ATTACK - reward good positioning when not attacking
    if not player_attacking:
        if current_distance <= 3.0:
            reward += 0.5
        elif current_distance <= 5.0:
            reward += 0.2
        else:
            reward -= 0.1

    return reward * env.dt

def unified_attack_effectiveness_reward(
    env: WarehouseBrawl,
    max_effective_range: float = 3.5,
    vulnerable_states: List[int] = None,
    close_range: float = 1.5,
    mid_range: float = 3.0,
    weapon_values: dict = None
) -> float:
    """
    Unified attack effectiveness reward that combines:
    - Effective attack reward (rewards hits, penalizes whiffs)
    - Attack frequency reward (only attack when in range)
    - Attack timing reward (attack when opponent is vulnerable)
    - Range appropriate attack reward (use right attacks for distance)
    - Weapon preference reward (use better weapons)

    Args:
        env (WarehouseBrawl): The game environment.
        max_effective_range (float): Maximum range where attacks can be effective.
        vulnerable_states (List[int]): List of opponent state IDs that are vulnerable.
        close_range (float): Distance threshold for close-range attacks.
        mid_range (float): Distance threshold for mid-range attacks.
        weapon_values (dict): Reward values for each weapon type.

    Returns:
        float: The computed unified attack effectiveness reward.
    """
    if vulnerable_states is None:
        vulnerable_states = [5, 6, 11]  # StunState, InAirState, KOState
    if weapon_values is None:
        weapon_values = {"Hammer": 2.0, "Spear": 1.5, "Punch": 0.0}

    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    player_attacking = isinstance(player.state, AttackState)

    if not player_attacking:
        return 0.0

    reward = 0.0
    distance = ((player.body.position.x - opponent.body.position.x)**2 +
                (player.body.position.y - opponent.body.position.y)**2)**0.5

    # 1. EFFECTIVE ATTACK - massive reward for successful hits
    if opponent.damage_taken_this_frame > 0:
        reward += 5.0

    # 2. RANGE CHECK - penalize attacks when too far
    if distance > 4.0:  # Too far for any attack to land
        reward -= 1.0
        return reward * env.dt  # Early return for bad positioning

    # 3. ATTACK FREQUENCY - only reward attacks when in effective range
    if distance <= max_effective_range:
        reward += 1.0
    else:
        reward -= 0.5

    # 4. ATTACK TIMING - reward attacking vulnerable opponents
    opponent_state_id = opponent.state_mapping.get(opponent.state.__class__.__name__, -1)
    if opponent_state_id in vulnerable_states:
        reward += 2.0
    else:
        reward += 0.5

    # 5. RANGE APPROPRIATE ATTACKS - reward using right attack for distance
    move_type = getattr(player.state, 'move_type', None)
    if move_type is not None:
        if distance <= close_range:
            # Close range: prefer light attacks
            if move_type in [MoveType.NLIGHT, MoveType.DLIGHT, MoveType.SLIGHT]:
                reward += 1.0
        elif distance <= mid_range:
            # Mid range: prefer signature attacks and aerials
            if move_type in [MoveType.NSIG, MoveType.DSIG, MoveType.SSIG, MoveType.SAIR]:
                reward += 1.0
        else:
            # Long range: prefer recovery and groundpound attacks
            if move_type in [MoveType.RECOVERY, MoveType.GROUNDPOUND]:
                reward += 1.0

    # 6. WEAPON PREFERENCE - reward using better weapons
    if distance <= max_effective_range:
        weapon_reward = weapon_values.get(player.weapon, 0.0)
        reward += weapon_reward

    return reward * env.dt

def attack_timing_reward(
    env: WarehouseBrawl,
    vulnerable_states: List[int] = None
) -> float:
    """
    Rewards well-timed attacks when opponent is in vulnerable states.
    Encourages strategic timing over random button mashing.

    Args:
        env (WarehouseBrawl): The game environment.
        vulnerable_states (List[int]): List of opponent state IDs that are vulnerable.

    Returns:
        float: The computed attack timing reward.
    """
    if vulnerable_states is None:
        vulnerable_states = [5, 6, 11]  # StunState, InAirState, KOState

    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    opponent_state_id = opponent.state_mapping.get(opponent.state.__class__.__name__, -1)
    player_attacking = isinstance(player.state, AttackState)

    if player_attacking and opponent_state_id in vulnerable_states:
        return 2.0 * env.dt
    elif player_attacking:
        return 0.5 * env.dt

    return 0.0

def weapon_preference_reward(
    env: WarehouseBrawl,
    weapon_values: dict = None,
    max_effective_range: float = 3.5
) -> float:
    """
    Rewards using stronger weapons only when attacking within effective range.
    Encourages weapon acquisition but prevents mindless attack spamming.

    Args:
        env (WarehouseBrawl): The game environment.
        weapon_values (dict): Reward values for each weapon type.
        max_effective_range (float): Maximum range where attacks can be effective.

    Returns:
        float: The computed weapon preference reward.
    """
    if weapon_values is None:
        weapon_values = {"Hammer": 2.0, "Spear": 1.5, "Punch": 0.0}

    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    if not isinstance(player.state, AttackState):
        return 0.0

    distance = ((player.body.position.x - opponent.body.position.x)**2 +
                (player.body.position.y - opponent.body.position.y)**2)**0.5

    # Only reward weapon usage when in effective range
    if distance <= max_effective_range:
        weapon_reward = weapon_values.get(player.weapon, 0.0)
        return weapon_reward * env.dt

    return 0.0

def combo_initiation_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Rewards starting attack sequences and maintaining offensive momentum.
    Encourages aggressive follow-ups after successful hits.

    Args:
        env (WarehouseBrawl): The game environment.

    Returns:
        float: The computed combo initiation reward.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    player_attacking = isinstance(player.state, AttackState)
    opponent_stunned = opponent.state_mapping.get(opponent.state.__class__.__name__, -1) == 5  # StunState

    if player_attacking and opponent_stunned:
        return 3.0 * env.dt
    elif player_attacking and opponent.damage_taken_this_frame > 0:
        return 1.5 * env.dt

    return 0.0

def unified_movement_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Unified movement reward that combines:
    - Offensive momentum reward (moving toward opponent)
    - Head to opponent reward (directional movement toward opponent)

    Args:
        env (WarehouseBrawl): The game environment.

    Returns:
        float: The computed unified movement reward.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Calculate movement toward opponent
    opponent_direction = 1 if opponent.body.position.x > player.body.position.x else -1
    player_movement = player.body.position.x - player.prev_x

    reward = 0.0

    # 1. OFFENSIVE MOMENTUM - reward moving toward opponent, penalize moving away
    if player_movement * opponent_direction > 0:
        reward += 1.0
    elif player_movement * opponent_direction < 0:
        reward -= 0.5

    # 2. HEAD TO OPPONENT - simple directional reward
    reward += player_movement * opponent_direction

    return reward * env.dt



def death_penalty_reward(
    env: WarehouseBrawl,
    death_penalty: float = 50.0
) -> float:
    """
    Applies a significant penalty when the player loses a stock (dies).
    Encourages defensive play and survival.

    Args:
        env (WarehouseBrawl): The game environment.
        death_penalty (float): Penalty applied when losing a life.

    Returns:
        float: The computed death penalty (negative reward).
    """
    player: Player = env.objects["player"]

    # Initialize stock tracking if it doesn't exist
    if not hasattr(player, 'prev_stock_count'):
        player.prev_stock_count = player.stocks

    # Check if player lost a stock this frame
    if player.stocks < player.prev_stock_count:
        player.prev_stock_count = player.stocks
        return -death_penalty * env.dt

    # Update stock count for next frame
    player.prev_stock_count = player.stocks
    return 0.0

def unified_input_control_reward(
    env: WarehouseBrawl,
    max_inputs_per_window: int = 4,
    window_size: int = 6,
    penalty_per_excess: float = 2.0,
    opposing_pairs: List[Tuple[str, str]] = None,
    penalty_scaling: float = 10.0,
    max_frames_memory: int = 5,
    consistency_bonus: float = 2.0,
    min_frames: int = 3
) -> float:
    """
    Unified input control reward that combines:
    - Input spam penalty (penalize excessive button frequency)
    - Button mashing penalty (penalize rapid alternating opposing actions)
    - Directional consistency reward (reward smooth movement)

    Args:
        env (WarehouseBrawl): The game environment.
        max_inputs_per_window (int): Maximum allowed inputs in the window.
        window_size (int): Size of the time window in frames.
        penalty_per_excess (float): Penalty for each input above the limit.
        opposing_pairs (List[Tuple[str, str]]): Pairs of opposing actions.
        penalty_scaling (float): How severely to penalize button mashing.
        max_frames_memory (int): How many frames to remember for detecting patterns.
        consistency_bonus (float): Reward for consistent movement.
        min_frames (int): Minimum frames of consistency required for reward.

    Returns:
        float: The computed unified input control reward.
    """
    if opposing_pairs is None:
        opposing_pairs = [("A", "D"), ("W", "S"), ("j", "k")]  # left-right, up-down, light-heavy

    player: Player = env.objects["player"]
    reward = 0.0

    # Initialize tracking if it doesn't exist
    if not hasattr(player, 'input_count_history'):
        player.input_count_history = []
    if not hasattr(player, 'action_history'):
        player.action_history = []
    if not hasattr(player, 'direction_history'):
        player.direction_history = []

    # 1. INPUT SPAM PENALTY
    current_inputs = sum(1 for key in ["W", "A", "S", "D", "space", "h", "l", "j", "k", "g"]
                        if player.input.key_status[key].just_pressed)

    player.input_count_history.append(current_inputs)
    if len(player.input_count_history) > window_size:
        player.input_count_history.pop(0)

    total_inputs = sum(player.input_count_history)
    if total_inputs > max_inputs_per_window:
        excess_inputs = total_inputs - max_inputs_per_window
        reward -= excess_inputs * penalty_per_excess

    # 2. BUTTON MASHING PENALTY
    current_action = {key: player.input.key_status[key].just_pressed for key in ["W", "A", "S", "D", "j", "k"]}
    player.action_history.append(current_action)

    if len(player.action_history) > max_frames_memory:
        player.action_history.pop(0)

    mashing_penalty = 0.0
    for key1, key2 in opposing_pairs:
        for i in range(1, min(len(player.action_history), max_frames_memory)):
            prev_frame = player.action_history[-i-1]
            curr_frame = player.action_history[-i]

            if (prev_frame.get(key1, False) and curr_frame.get(key2, False)) or \
               (prev_frame.get(key2, False) and curr_frame.get(key1, False)):
                time_penalty = penalty_scaling / max(i, 1)
                mashing_penalty += time_penalty

    mashing_penalty = min(mashing_penalty, 5.0)
    reward -= mashing_penalty

    # 3. DIRECTIONAL CONSISTENCY REWARD
    current_direction = 0
    if player.input.key_status["A"].held:
        current_direction = -1
    elif player.input.key_status["D"].held:
        current_direction = 1

    player.direction_history.append(current_direction)
    if len(player.direction_history) > 10:
        player.direction_history.pop(0)

    if len(player.direction_history) >= min_frames:
        recent_directions = player.direction_history[-min_frames:]
        if len(set(recent_directions)) == 1 and recent_directions[0] != 0:
            reward += consistency_bonus

    return reward * env.dt



def enhanced_survival_reward(
    env: WarehouseBrawl,
    base_reward: float = 2.0,
    time_scaling: float = 0.01,
    stock_bonus: float = 5.0,
    boundary_penalty: float = 10.0,
    platform_bonus: float = 1.0
) -> float:
    """
    Enhanced survival reward that includes:
    - Basic survival time reward
    - Boundary constraint (heavy penalty for going out of bounds)
    - Platform awareness (bonus for being on safe ground)

    Game boundaries and coordinate system:
    - Coordinate system: X increases right, Y increases DOWNWARD
    - Stage width: 29.8 tiles (±14.9)
    - Stage height: 16.8 tiles (±8.4, but Y+ is down, Y- is up)
    - Ground platforms: (4.5, 1) and (-4.5, 3) with width 10
    - Moving platform: (0, 1) with width 2
    """
    player: Player = env.objects["player"]

    # Initialize survival tracking if it doesn't exist
    if not hasattr(player, 'survival_start_time'):
        player.survival_start_time = env.steps
        player.last_stock_count = player.stocks

    reward = 0.0

    # 1. BOUNDARY CONSTRAINTS - heavy penalty for being near or outside boundaries
    x_pos = player.body.position.x
    y_pos = player.body.position.y

    # X boundaries: ±14.9 (stage width)
    x_danger_zone = 12.0  # Start penalizing before the actual boundary
    if abs(x_pos) > x_danger_zone:
        boundary_distance = abs(x_pos) - x_danger_zone
        reward -= boundary_distance * boundary_penalty

    # Y boundaries: ±8.4 (stage height) - but Y increases DOWNWARD
    # So positive Y means going down (falling), negative Y means going up (off-screen top)
    # We mainly care about the bottom boundary (positive Y values)
    y_danger_zone_bottom = 6.0  # Start penalizing when falling too low
    y_danger_zone_top = -6.0    # Also penalize going too high (negative Y)

    if y_pos > y_danger_zone_bottom:  # Falling too low
        boundary_distance = y_pos - y_danger_zone_bottom
        reward -= boundary_distance * boundary_penalty
    elif y_pos < y_danger_zone_top:  # Going too high (off-screen top)
        boundary_distance = y_danger_zone_top - y_pos
        reward -= boundary_distance * boundary_penalty

    # 2. PLATFORM AWARENESS - bonus for being on safe ground
    # NOTE: Y increases DOWNWARD in this coordinate system
    # Ground 1: center at (4.5, 1), width 10, so x range [-0.5, 9.5], y around 1
    # Ground 2: center at (-4.5, 3), width 10, so x range [-9.5, 0.5], y around 3
    # Moving platform: center at (0, 1), width 2, so x range [-1, 1], y around 1

    # Allow some tolerance for being "on" the platform (±0.5 in Y direction)
    on_ground1 = (-0.5 <= x_pos <= 9.5) and (0.5 <= y_pos <= 1.5)
    on_ground2 = (-9.5 <= x_pos <= 0.5) and (2.5 <= y_pos <= 3.5)
    on_moving_platform = (-1.0 <= x_pos <= 1.0) and (0.5 <= y_pos <= 1.5)

    if on_ground1 or on_ground2 or on_moving_platform:
        reward += platform_bonus
    else:
        # Small penalty for being off platform (encourages getting back to safety)
        reward -= 0.1

    # 3. STOCK LOSS DETECTION
    if player.stocks < player.last_stock_count:
        # Player died, reset survival timer but give bonus for time survived
        time_survived = env.steps - player.survival_start_time
        survival_bonus = (time_survived / env.fps) * time_scaling
        player.survival_start_time = env.steps  # Reset timer
        player.last_stock_count = player.stocks
        reward += survival_bonus

    # 4. SURVIVAL TIME REWARD
    current_survival_time = env.steps - player.survival_start_time

    # Base survival reward
    reward += base_reward

    # Time-scaled bonus
    time_bonus = (current_survival_time / env.fps) * time_scaling
    reward += time_bonus

    # Stock bonus
    stock_multiplier = 1.0 + (player.stocks - 1) * stock_bonus
    reward *= max(stock_multiplier, 0.1)  # Ensure it doesn't go negative

    return reward * env.dt

def death_penalty_reward(
    env: WarehouseBrawl,
    death_penalty: float = 50.0  # RESTORED to original
) -> float:
    """
    Applies a significant penalty when the player loses a stock (dies).
    Encourages defensive play and survival.

    Args:
        env (WarehouseBrawl): The game environment.
        death_penalty (float): Penalty applied when losing a life.

    Returns:
        float: The computed death penalty (negative reward).
    """
    player: Player = env.objects["player"]

    # Initialize stock tracking if it doesn't exist
    if not hasattr(player, 'prev_stock_count'):
        player.prev_stock_count = player.stocks

    # Check if player lost a stock this frame
    if player.stocks < player.prev_stock_count:
        player.prev_stock_count = player.stocks
        return -death_penalty * env.dt

    # Update stock count for next frame
    player.prev_stock_count = player.stocks
    return 0.0

def boundary_penalty_reward(
    env: WarehouseBrawl,
    boundary_penalty: float = 15.0,
    x_danger_zone: float = 12.0,
    y_danger_zone_bottom: float = 6.0,
    y_danger_zone_top: float = -6.0
) -> float:
    """
    Dedicated boundary penalty function that heavily penalizes going near or outside game boundaries.

    Game coordinate system and boundaries:
    - Coordinate system: X increases right, Y increases DOWNWARD
    - Stage width: 29.8 tiles, so X boundaries are approximately ±14.9
    - Stage height: 16.8 tiles, so Y boundaries are approximately ±8.4
    - Y+ is downward (falling), Y- is upward (off-screen top)

    Args:
        env (WarehouseBrawl): The game environment.
        boundary_penalty (float): Penalty scaling factor for boundary violations.
        x_danger_zone (float): X distance from center before penalty starts (±value).
        y_danger_zone_bottom (float): Y value below which penalty starts (falling).
        y_danger_zone_top (float): Y value above which penalty starts (going off-screen top).

    Returns:
        float: The computed boundary penalty (negative reward).
    """
    player: Player = env.objects["player"]

    x_pos = player.body.position.x
    y_pos = player.body.position.y

    penalty = 0.0

    # X boundaries: ±14.9 (stage width), start penalizing at x_danger_zone
    if abs(x_pos) > x_danger_zone:
        boundary_distance = abs(x_pos) - x_danger_zone
        penalty += boundary_distance * boundary_penalty

    # Y boundaries - remember Y+ is downward in this coordinate system
    if y_pos > y_danger_zone_bottom:  # Falling too low (positive Y)
        boundary_distance = y_pos - y_danger_zone_bottom
        penalty += boundary_distance * boundary_penalty
    elif y_pos < y_danger_zone_top:  # Going too high (negative Y, off-screen top)
        boundary_distance = y_danger_zone_top - y_pos
        penalty += boundary_distance * boundary_penalty

    return -penalty * env.dt

def platform_awareness_reward(
    env: WarehouseBrawl,
    platform_bonus: float = 1.0
) -> float:
    """
    Rewards staying on safe platforms and penalizes being in the air too long.

    Platform locations (accounting for Y+ downward coordinate system):
    - Ground 1: center at (4.5, 1), width 10, so x range [-0.5, 9.5], y around 1
    - Ground 2: center at (-4.5, 3), width 10, so x range [-9.5, 0.5], y around 3
    - Moving platform: center at (0, 1), width 2, so x range [-1, 1], y around 1

    Args:
        env (WarehouseBrawl): The game environment.
        platform_bonus (float): Reward for being on a safe platform.

    Returns:
        float: The computed platform awareness reward.
    """
    player: Player = env.objects["player"]

    x_pos = player.body.position.x
    y_pos = player.body.position.y

    # Check if player is on any platform (with tolerance for Y position)
    on_ground1 = (-0.5 <= x_pos <= 9.5) and (0.5 <= y_pos <= 1.5)
    on_ground2 = (-9.5 <= x_pos <= 0.5) and (2.5 <= y_pos <= 3.5)
    on_moving_platform = (-1.0 <= x_pos <= 1.0) and (0.5 <= y_pos <= 1.5)

    if on_ground1 or on_ground2 or on_moving_platform:
        return platform_bonus * env.dt
    else:
        # Small penalty for being off platform (encourages getting back to safety)
        return -0.1 * env.dt

# -------------------------------------------------------------------------
# ----------------------------- REWARD MANAGER ---------------------------
# -------------------------------------------------------------------------

# SIMPLIFIED REWARD FUNCTIONS FOR PERFORMANCE
def simple_positioning_reward(env: WarehouseBrawl) -> float:
    """Simplified positioning reward that only encourages getting close to opponent."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    distance = ((player.body.position.x - opponent.body.position.x)**2 +
                (player.body.position.y - opponent.body.position.y)**2)**0.5

    # Simple distance-based reward
    if distance < 2.0:
        return 1.0 * env.dt
    elif distance < 4.0:
        return 0.5 * env.dt
    else:
        return -0.1 * env.dt

def simple_attack_reward(env: WarehouseBrawl) -> float:
    """Simplified attack reward that only rewards attacking when close."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    player_attacking = isinstance(player.state, AttackState)
    if not player_attacking:
        return 0.0

    distance = ((player.body.position.x - opponent.body.position.x)**2 +
                (player.body.position.y - opponent.body.position.y)**2)**0.5

    # Reward attacking when close, penalize when far
    if distance < 3.0:
        return 1.0 * env.dt
    else:
        return -0.5 * env.dt

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    # SIMPLIFIED REWARD FUNCTIONS FOR FASTER TRAINING
    reward_functions = {
        # CORE DAMAGE - Most important signal, highest weight
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=30.0, params={'mode': RewardMode.ASYMMETRIC_OFFENSIVE}),

        # SIMPLIFIED SURVIVAL - Basic death penalty only
        'death_penalty_reward': RewTerm(func=death_penalty_reward, weight=10.0),

        # BOUNDARY SAFETY - Prevent going out of bounds
        'boundary_penalty_reward': RewTerm(func=boundary_penalty_reward, weight=5.0),

        # PLATFORM AWARENESS - Stay on safe ground
        'platform_awareness_reward': RewTerm(func=platform_awareness_reward, weight=2.0),

        # SIMPLIFIED POSITIONING - Only close-range engagement
        'positioning_reward': RewTerm(func=simple_positioning_reward, weight=5.0),

        # SIMPLIFIED ATTACK - Only attack when close
        'attack_reward': RewTerm(func=simple_attack_reward, weight=8.0),

        # BASIC CONSTRAINTS - Minimal overhead
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.1),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=100)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=30)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=15)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    # my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # OPTION 1: Start fresh with optimized RecurrentPPO for performance testing
    # my_agent = RecurrentPPOAgent()

    # OPTION 2: Load RecurrentPPO checkpoint (when available)
    # my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_optimized/rl_model_50001_steps.zip')

    # Note: Cannot load regular PPO checkpoints with RecurrentPPO due to LSTM architecture differences
    # If you want to continue from regular PPO, use SB3Agent instead:
    my_agent = SB3Agent(sb3_class=PPO, file_path='checkpoints/experiment_optimized/rl_model_10300206_steps')

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=50_000, # Reduced save frequency for faster checkpoints
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='experiment_optimized',  # New experiment name for optimized training
        mode=SaveHandlerMode.RESUME # FORCE to start a fresh optimized experiment
    )

    # Set opponent settings here:
    # REDUCED opponent variety for faster training
    opponent_specification = {
                    'self_play': (3.0, selfplay_handler),  # Reduced self-play weight
                    'constant_agent': (2.0, partial(ConstantAgent)),  # Increased simple agent weight
                    'based_agent': (5.0, partial(BasedAgent)),  # Increased simple agent weight
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=1_000_000_000,
        train_logging=TrainLogging.PLOT
    )

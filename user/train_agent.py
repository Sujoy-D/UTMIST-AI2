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
from stable_baselines3.common.callbacks import BaseCallback

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
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,
                # Standard deviation bounds for action distribution
                'log_std_init': -0.5,  # Initial log std dev (exp(-0.5) ≈ 0.6)
                'use_expln': True,     # Use exponential linear unit for std dev
                # Note: clip_mean is not supported by RecurrentActorCriticPolicy
            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=100000000,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs,
                                      # Additional std dev control parameters
                                      normalize_advantage=True,
                                      max_grad_norm=0.5)  # Gradient clipping for stability
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def clamp_policy_std(self, min_std: float = 0.1, max_std: float = 1.0) -> None:
        """
        Manually clamp the policy's standard deviation to prevent extreme values.

        Args:
            min_std: Minimum allowed standard deviation
            max_std: Maximum allowed standard deviation
        """
        # Clamp log std to achieve desired std bounds
        min_log_std = np.log(min_std)
        max_log_std = np.log(max_std)

        with torch.no_grad():
            # For RecurrentPPO with continuous actions, log_std is directly on the policy
            if hasattr(self.model.policy, 'log_std') and hasattr(self.model.policy.log_std, 'data'):
                old_log_std = self.model.policy.log_std.data.clone()
                self.model.policy.log_std.data.clamp_(min_log_std, max_log_std)
                return

            # Fallback: check action_net if it exists
            if hasattr(self.model.policy, 'action_net') and hasattr(self.model.policy.action_net, 'log_std'):
                if hasattr(self.model.policy.action_net.log_std, 'data'):
                    self.model.policy.action_net.log_std.data.clamp_(min_log_std, max_log_std)
                    return

    def get_current_std(self) -> float:
        """Get the current policy standard deviation for monitoring."""
        try:
            # For RecurrentPPO with continuous actions, log_std is directly on the policy
            if hasattr(self.model.policy, 'log_std') and hasattr(self.model.policy.log_std, 'data'):
                current_log_std = self.model.policy.log_std.data.mean().item()
                return np.exp(current_log_std)

            # Fallback: check action_net
            if hasattr(self.model.policy, 'action_net') and hasattr(self.model.policy.action_net, 'log_std'):
                if hasattr(self.model.policy.action_net.log_std, 'data'):
                    current_log_std = self.model.policy.action_net.log_std.data.mean().item()
                    return np.exp(current_log_std)
        except Exception as e:
            pass
        return float('nan')

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def get_training_stats(self) -> dict:
        """
        Get current training statistics similar to standard PPO output.
        Returns a dictionary with key training metrics.
        """
        stats = {}

        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Get episode statistics
            if hasattr(self.model.logger, 'name_to_value'):
                logger_dict = self.model.logger.name_to_value

                # Rollout statistics
                stats['ep_len_mean'] = logger_dict.get('rollout/ep_len_mean', 'N/A')
                stats['ep_rew_mean'] = logger_dict.get('rollout/ep_rew_mean', 'N/A')

                # Time statistics
                stats['fps'] = logger_dict.get('time/fps', 'N/A')
                stats['iterations'] = logger_dict.get('time/iterations', 'N/A')
                stats['time_elapsed'] = logger_dict.get('time/time_elapsed', 'N/A')
                stats['total_timesteps'] = logger_dict.get('time/total_timesteps', 'N/A')

                # Training statistics
                stats['approx_kl'] = logger_dict.get('train/approx_kl', 'N/A')
                stats['clip_fraction'] = logger_dict.get('train/clip_fraction', 'N/A')
                stats['clip_range'] = logger_dict.get('train/clip_range', 'N/A')
                stats['entropy_loss'] = logger_dict.get('train/entropy_loss', 'N/A')
                stats['explained_variance'] = logger_dict.get('train/explained_variance', 'N/A')
                stats['learning_rate'] = logger_dict.get('train/learning_rate', 'N/A')
                stats['loss'] = logger_dict.get('train/loss', 'N/A')
                stats['n_updates'] = logger_dict.get('train/n_updates', 'N/A')
                stats['policy_gradient_loss'] = logger_dict.get('train/policy_gradient_loss', 'N/A')
                stats['std'] = logger_dict.get('train/std', 'N/A')
                stats['value_loss'] = logger_dict.get('train/value_loss', 'N/A')

        # Get current policy std dev manually if not in logger
        if stats.get('std', 'N/A') == 'N/A':
            try:
                if hasattr(self.model.policy, 'log_std') and hasattr(self.model.policy.log_std, 'data'):
                    current_log_std = self.model.policy.log_std.data.mean().item()
                    stats['std'] = np.exp(current_log_std)
            except:
                stats['std'] = 'N/A'

        return stats

    def print_training_stats(self):
        """Print training statistics in a formatted table similar to SB3 output."""
        stats = self.get_training_stats()

        print("-" * 45)
        print("| rollout/                |             |")
        print(f"|    ep_len_mean          | {stats['ep_len_mean']:<11} |")
        print(f"|    ep_rew_mean          | {stats['ep_rew_mean']:<11} |")
        print("| time/                   |             |")
        print(f"|    fps                  | {stats['fps']:<11} |")
        print(f"|    iterations           | {stats['iterations']:<11} |")
        print(f"|    time_elapsed         | {stats['time_elapsed']:<11} |")
        print(f"|    total_timesteps      | {stats['total_timesteps']:<11} |")
        print("| train/                  |             |")
        print(f"|    approx_kl            | {stats['approx_kl']:<11} |")
        print(f"|    clip_fraction        | {stats['clip_fraction']:<11} |")
        print(f"|    clip_range           | {stats['clip_range']:<11} |")
        print(f"|    entropy_loss         | {stats['entropy_loss']:<11} |")
        print(f"|    explained_variance   | {stats['explained_variance']:<11} |")
        print(f"|    learning_rate        | {stats['learning_rate']:<11} |")
        print(f"|    loss                 | {stats['loss']:<11} |")
        print(f"|    n_updates            | {stats['n_updates']:<11} |")
        print(f"|    policy_gradient_loss | {stats['policy_gradient_loss']:<11} |")
        print(f"|    std                  | {stats['std']:<11} |")
        print(f"|    value_loss           | {stats['value_loss']:<11} |")
        print("-" * 45)

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0, clamp_std: bool = True, clamp_freq: int = 1000):
        self.model.set_env(env)
        self.model.verbose = verbose

        # Apply initial std dev clamping before training starts
        if clamp_std:
            self.clamp_policy_std(min_std=0.1, max_std=1.0)
            print("✓ Applied initial policy std dev clamping (0.1 - 1.0)")

        # Create callback for periodic std dev clamping
        callback = None
        if clamp_std:
            callback = StdDevClampingCallback(
                clamp_freq=clamp_freq,
                min_std=0.1,
                max_std=1.0,
                verbose=1 if verbose > 0 else 0
            )
            print(f"✓ Periodic std dev clamping enabled (every {clamp_freq} steps)")

        # Enable detailed logging for RecurrentPPO
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback,  # Use the proper callback
            tb_log_name="RecurrentPPO_run",  # TensorBoard logging name
            reset_num_timesteps=False  # Continue from existing timesteps if loading checkpoint
        )

        # Apply final std dev clamping after training
        if clamp_std:
            self.clamp_policy_std(min_std=0.1, max_std=1.0)
            print("✓ Applied final policy std dev clamping")

    def monitor_and_clamp_std(self, clamp: bool = True, min_std: float = 0.1, max_std: float = 1.0) -> None:
        """
        Monitor current std dev and optionally clamp it.
        Useful to call periodically during training.
        """
        current_std = self.get_current_std()
        if not np.isnan(current_std):
            print(f"Current policy std dev: {current_std:.6f}")
            if clamp and (current_std < min_std or current_std > max_std):
                print(f"  -> Clamping to range [{min_std}, {max_std}]")
                self.clamp_policy_std(min_std, max_std)
                new_std = self.get_current_std()
                print(f"  -> New std dev: {new_std:.6f}")
        else:
            print("Could not retrieve current policy std dev")

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
        # Jump if below map or opponent is above you
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

# REMOVED: Example reward functions - not used in current system

# REMOVED: head_to_middle_reward and head_to_opponent - functionality integrated into unified_movement_reward

# REMOVED: holding_more_than_3_keys - not used in simplified reward system

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
    """
    Reward for picking up weapons. Higher reward for better weapons.
    """
    if agent == "player":
        weapon = env.objects["player"].weapon
        if weapon == "Hammer":
            return 2.0  # Best weapon - highest reward
        elif weapon == "Spear":
            return 1.5  # Good weapon - medium reward
        else:
            return 0.5  # Any other weapon pickup (basic reward)
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    """
    Penalty for dropping weapons. Encourage keeping weapons equipped.
    """
    if agent == "player":
        return -1.0  # Always penalize dropping weapons
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

# REMOVED: unified_positioning_reward - functionality replaced by simple_positioning_reward for stability

# REMOVED: Complex attack system functions - replaced by simple_attack_reward and simple_damage_reward for stability

def unified_movement_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Unified movement reward that considers both X and Y movement toward opponent.
    Rewards movement that reduces distance to opponent in both dimensions.
    STABILITY: Added aggressive capping to prevent reward spikes.

    Game coordinate system:
    - X increases right (positive X = right, negative X = left)
    - Y increases DOWNWARD (positive Y = down/falling, negative Y = up/jumping)

    Args:
        env (WarehouseBrawl): The game environment.

    Returns:
        float: The computed unified movement reward.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Calculate current and previous distances to opponent
    current_pos = (player.body.position.x, player.body.position.y)
    prev_pos = (player.prev_x, player.prev_y)
    opponent_pos = (opponent.body.position.x, opponent.body.position.y)

    # Calculate distance changes
    current_distance = ((current_pos[0] - opponent_pos[0])**2 + (current_pos[1] - opponent_pos[1])**2)**0.5
    prev_distance = ((prev_pos[0] - opponent_pos[0])**2 + (prev_pos[1] - opponent_pos[1])**2)**0.5

    reward = 0.0

    # 1. OVERALL APPROACH REWARD - reward for getting closer overall
    distance_change = prev_distance - current_distance
    if distance_change > 0:  # Got closer
        reward += min(distance_change * 1.0, 0.3)  # Reduced cap to prevent spikes
    else:  # Got further or stayed same
        reward += max(distance_change * 0.25, -0.2)  # Reduced penalty

    # 2. DIRECTIONAL MOVEMENT REWARDS - reward movement in correct directions

    # X-direction movement
    x_movement = current_pos[0] - prev_pos[0]
    x_direction_to_opponent = 1 if opponent_pos[0] > current_pos[0] else -1

    if abs(opponent_pos[0] - current_pos[0]) > 0.5:  # Only care if there's meaningful X distance
        if x_movement * x_direction_to_opponent > 0:  # Moving toward opponent in X
            reward += 0.1  # Reduced from 0.3
        elif x_movement * x_direction_to_opponent < 0:  # Moving away from opponent in X
            reward -= 0.05  # Reduced from 0.2

    # Y-direction movement (remember Y+ is downward, Y- is upward)
    y_movement = current_pos[1] - prev_pos[1]
    y_direction_to_opponent = 1 if opponent_pos[1] > current_pos[1] else -1

    if abs(opponent_pos[1] - current_pos[1]) > 0.5:  # Only care if there's meaningful Y distance
        if y_movement * y_direction_to_opponent > 0:  # Moving toward opponent in Y
            reward += 0.1  # Reduced from 0.3
        elif y_movement * y_direction_to_opponent < 0:  # Moving away from opponent in Y
            reward -= 0.05  # Reduced from 0.2

    # 3. BONUS for simultaneous X and Y approach
    x_approaching = (abs(opponent_pos[0] - current_pos[0]) > 0.5) and (x_movement * x_direction_to_opponent > 0)
    y_approaching = (abs(opponent_pos[1] - current_pos[1]) > 0.5) and (y_movement * y_direction_to_opponent > 0)

    if x_approaching and y_approaching:
        reward += 0.2  # Reduced bonus from 0.5

    # FINAL SAFETY CAP - ensure no single reward exceeds bounds (ULTRA-CONSERVATIVE)
    reward = max(-0.1, min(0.1, reward))

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
        return death_penalty * env.dt

    # Update stock count for next frame
    player.prev_stock_count = player.stocks
    return 0.0

# REMOVED: Complex input control system - too complex and potentially unstable



# REMOVED: Complex survival system and redundant boundary/platform functions - replaced by precise functions

# -------------------------------------------------------------------------
# ----------------------------- REWARD MANAGER ---------------------------
# -------------------------------------------------------------------------

# SIMPLIFIED REWARD FUNCTIONS FOR PERFORMANCE AND STABILITY
def simple_positioning_reward(env: WarehouseBrawl) -> float:
    """Simple distance-based positioning reward. ULTRA-CONSERVATIVE."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Safe distance calculation with bounds checking
    dx = player.body.position.x - opponent.body.position.x
    dy = player.body.position.y - opponent.body.position.y
    distance = max(0.1, (dx*dx + dy*dy)**0.5)  # Prevent division by zero

    # Simple tiered reward (ULTRA-CONSERVATIVE values)
    if distance < 2.0:
        reward = 0.05 * env.dt
    elif distance < 4.0:
        reward = 0.02 * env.dt
    else:
        reward = -0.005 * env.dt

    # Safety cap
    return max(-0.02, min(0.02, reward))

def simple_attack_reward(env: WarehouseBrawl) -> float:
    """
    Simple attack reward that considers distance and attack direction.
    Rewards attacking when close AND facing toward opponent.
    Penalizes attacking when facing away from opponent.
    STABILITY: Added aggressive capping to prevent reward spikes.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    player_attacking = isinstance(player.state, AttackState)
    if not player_attacking:
        return 0.0

    # Safe distance calculation
    dx = player.body.position.x - opponent.body.position.x
    dy = player.body.position.y - opponent.body.position.y
    distance = (dx*dx + dy*dy)**0.5

    reward = 0.0

    # 1. DISTANCE CHECK - only meaningful rewards/penalties when reasonably close
    if distance > 4.0:  # Too far for any attack to be effective
        return -0.05 * env.dt  # Reduced penalty

    # 2. FACING DIRECTION CHECK - X axis (left/right)
    # opponent_direction: +1 if opponent is to the right, -1 if to the left
    opponent_x_direction = 1 if opponent.body.position.x > player.body.position.x else -1
    # player.facing.value: +1 if facing right, -1 if facing left
    facing_toward_opponent_x = (player.facing.value * opponent_x_direction) > 0

    # 3. VERTICAL ALIGNMENT CHECK - Y axis (up/down)
    # For attacks to be effective, player should be at similar or appropriate Y level
    y_difference = abs(player.body.position.y - opponent.body.position.y)
    good_y_alignment = y_difference < 2.0  # Within reasonable vertical range

    # 4. REWARD STRUCTURE based on distance and direction (REDUCED VALUES)
    if distance < 2.0:  # Very close range
        if facing_toward_opponent_x and good_y_alignment:
            reward += 0.2  # Reduced from 0.5
        elif facing_toward_opponent_x:
            reward += 0.1  # Reduced from 0.3
        elif good_y_alignment:
            reward += 0.05  # Reduced from 0.1
        else:
            reward -= 0.05  # Reduced from 0.2

    elif distance < 3.0:  # Medium close range
        if facing_toward_opponent_x and good_y_alignment:
            reward += 0.1  # Reduced from 0.3
        elif facing_toward_opponent_x:
            reward += 0.05  # Reduced from 0.2
        else:
            reward -= 0.02  # Reduced from 0.1

    else:  # Far range (3.0 - 4.0)
        if facing_toward_opponent_x and good_y_alignment:
            reward += 0.02  # Reduced from 0.1
        else:
            reward -= 0.02  # Reduced from 0.1

    # FINAL SAFETY CAP (ULTRA-CONSERVATIVE)
    reward = max(-0.05, min(0.05, reward))

    return reward * env.dt

def simple_damage_reward(env: WarehouseBrawl) -> float:
    """Simple damage-based reward - most important signal. ULTRA-CONSERVATIVE capping."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Only reward damage dealt, ignore damage taken for simplicity
    damage_dealt = opponent.damage_taken_this_frame

    # Normalize and cap the reward to prevent spikes (ULTRA-CONSERVATIVE)
    reward = min(damage_dealt / 140.0, 0.1) * env.dt  # Much smaller cap
    return max(-0.05, min(0.05, reward))  # Additional safety cap

def precise_boundary_penalty(env: WarehouseBrawl) -> float:
    """Precise boundary penalty using actual environment coordinate system."""
    player: Player = env.objects["player"]

    x_pos = player.body.position.x
    y_pos = player.body.position.y

    penalty = 0.0

    # Use exact environment boundaries from notebook documentation:
    # Screen dimensions: width 14.9, height 9.94
    # Origin (0,0) at center, so bounds are ±7.45 in X, ±4.97 in Y
    screen_half_width = 14.9 / 2   # ±7.45
    screen_half_height = 9.94 / 2  # ±4.97

    # X boundaries - screen goes from -7.45 to +7.45
    left_danger_zone = -screen_half_width + 1.5  # Start penalty 1.5 units from left edge (-5.95)
    right_danger_zone = screen_half_width - 1.5  # Start penalty 1.5 units from right edge (5.95)

    if x_pos < left_danger_zone:  # Going off-screen to the LEFT
        edge_distance = left_danger_zone - x_pos
        penalty += edge_distance * 0.5  # Progressive penalty
    elif x_pos > right_danger_zone:  # Going off-screen to the RIGHT
        edge_distance = x_pos - right_danger_zone
        penalty += edge_distance * 0.5  # Progressive penalty

    # Y boundaries - screen extends from -4.97 to +4.97 (positive Y goes DOWN)
    # Bottom boundary (falling off) - positive Y values (falling down)
    bottom_danger_zone = screen_half_height - 1.0  # Start penalty 1.0 unit from bottom (3.97)
    if y_pos > bottom_danger_zone:
        fall_distance = y_pos - bottom_danger_zone
        penalty += fall_distance * 1.5  # Stronger penalty for falling

    # Top boundary (going off-screen up) - negative Y values (going up)
    top_danger_zone = -screen_half_height + 1.0  # Start penalty 1.0 unit from top (-3.97)
    if y_pos < top_danger_zone:
        edge_distance = top_danger_zone - y_pos
        penalty += edge_distance * 0.3  # Lower penalty for going up

    return -penalty * env.dt

def precise_platform_reward(env: WarehouseBrawl) -> float:
    """Precise platform reward using actual environment platform coordinates."""
    player: Player = env.objects["player"]

    x_pos = player.body.position.x
    y_pos = player.body.position.y

    reward = 0.0

    # From notebook documentation:
    # Ground1: y=2.85 between x=-7.0 and x=-2.0 (width: 4.15)
    # Ground2: y=0.85 between x=2.0 and x=7.0 (width: 4.15)

    # Ground 1: x range [-7.0, -2.0], y around 2.85
    on_ground1 = (-7.0 <= x_pos <= -2.0) and (2.5 <= y_pos <= 3.2)

    # Ground 2: x range [2.0, 7.0], y around 0.85
    on_ground2 = (2.0 <= x_pos <= 7.0) and (0.5 <= y_pos <= 1.2)

    # Moving platform: Get actual coordinates from environment
    if 'platform1' in env.objects:
        platform = env.objects['platform1']
        platform_x = platform.body.position.x
        platform_y = platform.body.position.y
        platform_half_width = platform.width / 2  # From Stage class: width=2, so half_width=1.0

        # Platform detection with some tolerance
        platform_x_range = 0.5  # Additional tolerance
        platform_y_range = 0.3  # Additional tolerance

        on_moving_platform = (
            (platform_x - platform_half_width - platform_x_range <= x_pos <= platform_x + platform_half_width + platform_x_range) and
            (platform_y - platform_y_range <= y_pos <= platform_y + platform.height + platform_y_range)
        )
    else:
        # Fallback to hardcoded values if platform1 doesn't exist
        on_moving_platform = (-1.5 <= x_pos <= 1.5) and (-0.2 <= y_pos <= 2.2)

    if on_ground1 or on_ground2:
        reward += 0.3  # Main platforms
    elif on_moving_platform:
        reward += 0.5  # Bonus for using moving platform (more skill required)
    else:
        # Small penalty for being off any platform
        reward -= 0.02

    return reward * env.dt

# REMOVED: camera_bounds_penalty - functionality integrated into precise_boundary_penalty

def simple_survival_reward(env: WarehouseBrawl) -> float:
    """Simple survival reward - just being alive."""
    return 0.01 * env.dt


def weapon_approach_reward(env: WarehouseBrawl) -> float:
    """
    Optimized reward for approaching spawned weapons on the ground.
    Encourages strategic weapon collection behavior.

    Args:
        env (WarehouseBrawl): The game environment.

    Returns:
        float: The computed weapon approach reward.
    """
    player: Player = env.objects["player"]

    # Early exit if player already has a good weapon
    if player.weapon in ["Hammer", "Spear"]:
        return 0.0

    # Cache player positions once
    player_pos = (player.body.position.x, player.body.position.y)
    player_prev_pos = (player.prev_x, player.prev_y)

    best_weapon_reward = 0.0
    closest_weapon_distance = float('inf')

    # Weapon type mappings for efficiency
    weapon_values = {
        'Hammer': 2.0,
        'Spear': 1.5,
        'hammer': 2.0,  # lowercase for pattern matching
        'spear': 1.5
    }

    # Single pass through objects with efficient checks
    for obj_name, obj in env.objects.items():
        # Skip non-weapon objects quickly
        if obj_name in ['player', 'opponent'] or not hasattr(obj, 'body'):
            continue

        # Efficient weapon detection
        weapon_value = 1.0  # Default
        is_weapon = False

        # Primary detection: check for weapon attributes
        if hasattr(obj, 'weapon_type'):
            weapon_value = weapon_values.get(obj.weapon_type, 1.0)
            is_weapon = True
        elif hasattr(obj, 'name'):
            weapon_value = weapon_values.get(obj.name, 1.0)
            is_weapon = obj.name in ['Hammer', 'Spear']

        # Fallback: pattern matching (only if needed and no weapons found yet)
        if not is_weapon and best_weapon_reward == 0.0:
            obj_name_lower = obj_name.lower()
            if any(pattern in obj_name_lower for pattern in ['weapon', 'hammer', 'spear']):
                weapon_value = weapon_values.get(obj_name_lower.split('_')[0], 1.0)  # Get first part
                is_weapon = True

        if not is_weapon:
            continue

        # Get weapon position
        try:
            weapon_pos = (obj.body.position.x, obj.body.position.y)
        except AttributeError:
            continue  # Skip if no valid position

        # Fast distance check (squared distance first to avoid sqrt)
        dx = player_pos[0] - weapon_pos[0]
        dy = player_pos[1] - weapon_pos[1]
        distance_squared = dx*dx + dy*dy

        # Early distance filtering (8.0^2 = 64.0)
        if distance_squared > 64.0:
            continue

        # Now calculate actual distances only for nearby weapons
        current_distance = distance_squared**0.5

        # Calculate previous distance
        dx_prev = player_prev_pos[0] - weapon_pos[0]
        dy_prev = player_prev_pos[1] - weapon_pos[1]
        prev_distance = (dx_prev*dx_prev + dy_prev*dy_prev)**0.5

        # Calculate approach reward
        distance_change = prev_distance - current_distance

        weapon_reward = 0.0
        if distance_change > 0:  # Moving closer
            weapon_reward = min(distance_change * weapon_value * 0.5, 0.3)
        elif distance_change < 0:  # Moving away
            weapon_reward = max(distance_change * weapon_value * 0.2, -0.1)

        # Track best opportunity
        if current_distance < closest_weapon_distance:
            closest_weapon_distance = current_distance
            best_weapon_reward = weapon_reward

            # Early termination for very close high-value weapons
            if current_distance < 2.0 and weapon_value >= 2.0:
                break

    return best_weapon_reward * env.dt

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
'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    # BALANCED REWARD SYSTEM - Optimized for stable training
    reward_functions = {
        # CORE MOVEMENT & POSITIONING (fundamental skills)
        'movement_reward': RewTerm(func=unified_movement_reward, weight=1.0),  # Base movement toward opponent
        'positioning_reward': RewTerm(func=simple_positioning_reward, weight=0.8),  # Distance-based positioning

        # CORE COMBAT (primary learning objectives)
        'attack_reward': RewTerm(func=simple_attack_reward, weight=2.0),  # Strategic attacking
        'damage_reward': RewTerm(func=simple_damage_reward, weight=3.0),  # Damage dealing (highest priority)

        # WEAPON COLLECTION (strategic enhancement)
        'weapon_approach_reward': RewTerm(func=weapon_approach_reward, weight=0.6),  # Weapon collection

        # ENVIRONMENT CONSTRAINTS (safety and boundaries)
        'boundary_penalty': RewTerm(func=precise_boundary_penalty, weight=1.5),  # Strong boundary enforcement
        'platform_reward': RewTerm(func=precise_platform_reward, weight=0.7),  # Platform usage

        # SURVIVAL (fundamental requirement)
        'death_penalty': RewTerm(func=death_penalty_reward, weight=-2.0),  # Strong death penalty

        # INPUT QUALITY (action efficiency)
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.3),  # Prevent button mashing
    }

    signal_subscriptions = {
        # MAJOR GAME EVENTS (ultimate objectives)
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=100.0)),  # Very high win reward
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=15.0)),  # Strong KO reward

        # TACTICAL EVENTS (skill development)
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=8.0)),  # Combo rewards

        # WEAPON MANAGEMENT (strategic play)
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=5.0)),  # Weapon pickup
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=5.0))  # Weapon drop penalty
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- CALLBACK CLASSES -------------------------
# -------------------------------------------------------------------------

class StdDevClampingCallback(BaseCallback):
    """
    Callback for periodically clamping the policy standard deviation.
    This is safe for pickling and won't cause serialization issues.
    """
    def __init__(self, clamp_freq: int = 1000, min_std: float = 0.1, max_std: float = 1.0, verbose: int = 0):
        super().__init__(verbose)
        self.clamp_freq = clamp_freq
        self.min_std = min_std
        self.max_std = max_std
        self.min_log_std = np.log(min_std)
        self.max_log_std = np.log(max_std)

    def _on_step(self) -> bool:
        """Called after every step. Returns True to continue training."""
        if self.n_calls % self.clamp_freq == 0:
            self._clamp_policy_std()
        return True

    def _clamp_policy_std(self) -> None:
        """Clamp the policy standard deviation."""
        model = self.model

        with torch.no_grad():
            # For RecurrentPPO with continuous actions, log_std is directly on the policy
            if hasattr(model.policy, 'log_std') and hasattr(model.policy.log_std, 'data'):
                old_log_std = model.policy.log_std.data.mean().item()
                model.policy.log_std.data.clamp_(self.min_log_std, self.max_log_std)
                new_log_std = model.policy.log_std.data.mean().item()

                if self.verbose > 0:
                    old_std = np.exp(old_log_std)
                    new_std = np.exp(new_log_std)
                    print(f"Step {self.n_calls}: Std dev {old_std:.6f} -> {new_std:.6f}")
                return

            # Fallback: check action_net if it exists
            if hasattr(model.policy, 'action_net') and hasattr(model.policy.action_net, 'log_std'):
                if hasattr(model.policy.action_net.log_std, 'data'):
                    old_log_std = model.policy.action_net.log_std.data.mean().item()
                    model.policy.action_net.log_std.data.clamp_(self.min_log_std, self.max_log_std)
                    new_log_std = model.policy.action_net.log_std.data.mean().item()

                    if self.verbose > 0:
                        old_std = np.exp(old_log_std)
                        new_std = np.exp(new_log_std)
                        print(f"Step {self.n_calls}: Std dev {old_std:.6f} -> {new_std:.6f}")
                return

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    # my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # OPTION 2: Load existing PPO checkpoint (if compatible)
    # my_agent = SB3Agent(sb3_class=PPO, file_path='checkpoints/experiment_11/rl_model_10300103_steps.zip')

    # If you want to use RecurrentPPO instead, start fresh:
    # my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_19/rl_model_4500045_steps')

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayLatest(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='experiment_19',
        mode=SaveHandlerMode.RESUME # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (8, selfplay_handler),
                    'constant_agent': (0.5, partial(ConstantAgent)),
                    'based_agent': (1.5, partial(BasedAgent)),
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

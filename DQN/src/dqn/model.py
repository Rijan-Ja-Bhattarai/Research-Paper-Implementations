import gymnasium as gym
import torch.nn as nn 
import numpy as np
from collections import defaultdict

class QNetwork(nn.Module):
    def __init__(self, 
                 env: gym.Env,
                 learning_rate: float, 
                 initial_epsilon: float, 
                 epsilon_decay: float, 
                 final_epsilon: float, 
                 discount_factor: float = 0.95):
        super().__init__()

        self.env = env 
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        # One temporal-difference error is recorded for every environment step.
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]):
        """
        Choose an action through epsilon-greedy strategy

        Actions:
            Stand = 0
            Hit = 1
        """

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self, 
            obs: tuple[int, int, bool],
            action: int, 
            reward: float, 
            terminated: bool, 
            next_obs: tuple[int, int, bool]
    ):
        """
        Update Q Values based on experience 
        """

        future_q_values = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_values
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.learning_rate * temporal_difference
        )

        # Track Learning Progress 
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

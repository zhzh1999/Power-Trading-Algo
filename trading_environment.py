# This script defines the custom Gym environment for power market trading.
# trading_environment.py

import gym
from gym import spaces
import numpy as np
import pandas as pd

class PowerTradingEnv(gym.Env):
    """
    Custom Environment for Power Market Trading
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=100000, window_size=24, features=None):
        super(PowerTradingEnv, self).__init__()

        self.data = data.reset_index()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for neutral
        self.max_steps = len(data) - window_size - 1
        self.features = features

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: window_size * features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size * len(self.features),), dtype=np.float32
        )

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = self.window_size
        return self._get_obs()

    def _get_obs(self):
        window_data = self.data.loc[self.current_step - self.window_size:self.current_step -1, self.features].values
        return window_data.flatten()

    def step(self, action):
        done = False
        reward = 0
        info = {}

        current_price = self.data.loc[self.current_step, 'price']

        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.balance -= current_price
            elif self.position == -1:
                self.position = 0
                self.balance -= current_price  # Cover short
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.balance += current_price
            elif self.position == 1:
                self.position = 0
                self.balance += current_price  # Sell long

        # Calculate reward
        if self.current_step + 1 < len(self.data):
            next_price = self.data.loc[self.current_step +1, 'price']
            price_change = next_price - current_price

            if self.position == 1:
                reward = price_change  # Profit if price increases
            elif self.position == -1:
                reward = -price_change  # Profit if price decreases
            else:
                reward = 0

            # Update balance with current position's P&L
            self.balance += self.position * price_change
        else:
            reward = 0

        # Move to next step
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, info

    def render(self, mode='human', close=False):
        profit = self.balance - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Profit: {profit:.2f}, Position: {self.position}")

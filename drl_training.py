# This script sets up and trains the Deep Reinforcement Learning (DRL) agent using the PPO algorithm.
# drl_training.py

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_environment import PowerTradingEnv

def load_data(filename='processed_data.csv'):
    data = pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
    return data

def main():
    # Load and prepare data
    data = load_data()
    features = [
        'price_lag1', 'price_lag24', 'price_lag168',
        'demand_lag1', 'demand_lag24', 'demand_lag168',
        'temperature_lag1', 'temperature_lag24', 'temperature_lag168',
        'wind_speed_lag1', 'wind_speed_lag24', 'wind_speed_lag168',
        'price_roll_mean_24', 'price_roll_std_24',
        'demand_roll_mean_24',
        'temperature_roll_mean_24', 'wind_speed_roll_mean_24',
        'hour', 'day_of_week', 'month'
    ]
    timesteps = 24

    # Initialize the environment
    env = PowerTradingEnv(data, initial_balance=100000, window_size=timesteps, features=features)
    env = DummyVecEnv([lambda: env])

    # Initialize the PPO agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("ppo_power_trading")

if __name__ == "__main__":
    main()

# This script evaluates the trained DRL agent by running it in the environment and plotting the portfolio balance over time.
# drl_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
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

    # Load the trained model
    model = PPO.load("ppo_power_trading")

    # Evaluation
    obs = env.reset()
    done = False
    balance_history = [env.balance]

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        balance_history.append(env.balance)

    # Plot the balance over time
    plt.figure(figsize=(14, 7))
    plt.plot(balance_history, label='Portfolio Balance')
    plt.xlabel('Time Step')
    plt.ylabel('Balance ($)')
    plt.title('DRL Trading Strategy Performance')
    plt.legend()
    plt.show()

    print(f"Final Portfolio Balance: ${env.balance:.2f}")

if __name__ == "__main__":
    main()

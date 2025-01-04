# This script handles data generation (synthetic in this case), feature engineering, and preprocessing.
# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data():
    date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)
    price = np.random.normal(loc=50, scale=10, size=len(date_rng))  # Synthetic prices in $/MWh
    demand = np.random.normal(loc=1000, scale=200, size=len(date_rng))  # Synthetic demand in MW
    temperature = np.random.normal(loc=15, scale=10, size=len(date_rng))  # Synthetic temperature in Celsius
    wind_speed = np.random.normal(loc=10, scale=3, size=len(date_rng))  # Synthetic wind speed in m/s

    data = pd.DataFrame({
        'timestamp': date_rng,
        'price': price,
        'demand': demand,
        'temperature': temperature,
        'wind_speed': wind_speed
    })
    data.set_index('timestamp', inplace=True)
    return data

def feature_engineering(data):
    # Feature Engineering
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month

    # Lag features
    for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
        data[f'price_lag{lag}'] = data['price'].shift(lag)
        data[f'demand_lag{lag}'] = data['demand'].shift(lag)
        data[f'temperature_lag{lag}'] = data['temperature'].shift(lag)
        data[f'wind_speed_lag{lag}'] = data['wind_speed'].shift(lag)

    # Rolling statistics
    data['price_roll_mean_24'] = data['price'].rolling(window=24).mean()
    data['price_roll_std_24'] = data['price'].rolling(window=24).std()
    data['demand_roll_mean_24'] = data['demand'].rolling(window=24).mean()
    data['temperature_roll_mean_24'] = data['temperature'].rolling(window=24).mean()
    data['wind_speed_roll_mean_24'] = data['wind_speed'].rolling(window=24).mean()

    # Drop rows with NaN values due to lagging and rolling
    data.dropna(inplace=True)
    return data

def save_processed_data(data, filename='processed_data.csv'):
    data.to_csv(filename)

def main():
    data = generate_synthetic_data()
    data = feature_engineering(data)
    save_processed_data(data)

if __name__ == "__main__":
    main()

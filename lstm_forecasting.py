# This script builds, trains, and evaluates the LSTM model for price forecasting.
# lstm_forecasting.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_data(filename='processed_data.csv'):
    data = pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
    return data

def prepare_lstm_data(data, features, target, timesteps=24):
    X = data[features].values
    y = data[target].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM: [samples, timesteps, features]
    X_lstm = []
    y_lstm = []
    for i in range(timesteps, len(X_scaled)):
        X_lstm.append(X_scaled[i-timesteps:i])
        y_lstm.append(y[i])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    # Split into training and testing sets
    split = int(0.8 * len(X_lstm))
    X_train, X_test = X_lstm[:split], X_lstm[split:]
    y_train, y_test = y_lstm[:split], y_lstm[split:]

    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('LSTM Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.show()

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.legend()
    plt.title('LSTM Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price ($/MWh)')
    plt.show()

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
    target = 'price'
    timesteps = 24

    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(data, features, target, timesteps)

    # Build and train the model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"LSTM Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot actual vs predicted prices
    plot_predictions(y_test, y_pred)

    # Save the model and scaler for later use
    model.save('lstm_price_forecast_model.h5')
    import joblib
    joblib.dump(scaler, 'scaler.save')

if __name__ == "__main__":
    main()

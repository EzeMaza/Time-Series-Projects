import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanAbsoluteError, R2Score

def build_and_train_lstm(X_train, y_train, activation='tanh', units1=64, units2=32, epochs=5, batch_size=32):
    model = Sequential([
        LSTM(units1, activation=activation, return_sequences=True),
        LSTM(units2, activation=activation),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(),
            R2Score()
        ])
    
    # Train model
    summary = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model, summary

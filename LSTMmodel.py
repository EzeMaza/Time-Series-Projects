import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanAbsoluteError, R2Score
from tensorflow.keras.optimizers import Adam

def build_and_train_lstm(X_train, y_train, activation='tanh', units1=64, units2=32, 
                         third_layer=False, units3=16, epochs=5, batch_size=32, 
                         learning_rate=1e-3):
    
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units1, activation=activation, return_sequences=True))
    
    # Second LSTM layer
    if third_layer:
        model.add(LSTM(units2, activation=activation, return_sequences=True))  # Keep return_sequences=True for third layer
        model.add(LSTM(units3, activation=activation))  # Third LSTM layer
    else:
        model.add(LSTM(units2, activation=activation))  # Second LSTM layer
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model with custom learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[MeanAbsoluteError(), R2Score()]
    )
    
    # Train model
    summary = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model, summary

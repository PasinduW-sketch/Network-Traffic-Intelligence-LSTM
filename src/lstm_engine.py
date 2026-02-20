"""
Improved Bidirectional LSTM for Network Traffic Forecasting
Enhanced architecture for better accuracy
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

class TrafficLSTM:
    def __init__(self, sequence_length=24, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, units=128, dropout_rate=0.2):
        """Build improved Bidirectional LSTM model"""
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(units, return_sequences=True), 
                         input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(units // 2, return_sequences=True)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Third LSTM layer
            LSTM(units // 4, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate / 2),
            
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model with callbacks"""
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                             factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'R2': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        return metrics, predictions
    
    def plot_training_history(self, save_path='results/training_history.png'):
        """Plot training history"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_title('Model MAE During Training', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_predictions(self, y_true, y_pred, save_path='results/predictions.png'):
        """Plot predictions vs actual"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        ax1.plot(y_true, label='Actual', linewidth=2, color='#2a5298')
        ax1.plot(y_pred, label='Predicted', linewidth=2, color='#28a745', linestyle='--')
        ax1.set_title('Network Traffic: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (Hours)')
        ax1.set_ylabel('Traffic (GB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.6, color='#2a5298', edgecolors='white', linewidth=0.5)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_title('Prediction Accuracy (R²)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Actual Traffic (GB)')
        ax2.set_ylabel('Predicted Traffic (GB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add R² text
        r2 = r2_score(y_true, y_pred)
        ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def save_model(self, path='models/lstm_model.h5'):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        return path


def generate_sample_data(n_samples=1000, noise_factor=0.1):
    """Generate realistic network traffic data"""
    np.random.seed(42)
    
    # Create daily pattern (24-hour cycle)
    hours = np.arange(n_samples)
    daily_pattern = 20 + 15 * np.sin(2 * np.pi * hours / 24 - np.pi/2)  # Peak at noon
    
    # Weekly pattern
    weekly_pattern = 5 * np.sin(2 * np.pi * hours / (24 * 7))
    
    # Trend
    trend = 0.01 * hours
    
    # Noise
    noise = noise_factor * np.random.randn(n_samples) * daily_pattern
    
    # Combine
    traffic = daily_pattern + weekly_pattern + trend + noise
    traffic = np.maximum(traffic, 0)  # No negative traffic
    
    return traffic


def train_and_evaluate():
    """Train and evaluate the improved model"""
    print("="*60)
    print("Network Traffic Intelligence - LSTM Model Training")
    print("="*60)
    
    # Generate data
    print("\n[1] Generating sample network traffic data...")
    data = generate_sample_data(n_samples=1500)
    
    # Initialize model
    model = TrafficLSTM(sequence_length=48, n_features=1)
    
    # Scale data
    data_scaled = model.scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = model.create_sequences(data_scaled)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Build model
    print("\n[2] Building improved Bidirectional LSTM model...")
    model.build_model(units=128, dropout_rate=0.2)
    model.model.summary()
    
    # Train
    print("\n[3] Training model...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate
    print("\n[4] Evaluating model...")
    metrics, predictions = model.evaluate(X_test, y_test)
    
    # Inverse transform
    y_test_inv = model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_inv = model.scaler.inverse_transform(predictions).flatten()
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"   R² Score:     {metrics['R2']:.4f} ({metrics['R2']*100:.2f}%)")
    print(f"   MAE:          {metrics['MAE']:.4f} GB")
    print(f"   RMSE:         {metrics['RMSE']:.4f} GB")
    print(f"   MSE:          {metrics['MSE']:.4f}")
    print("="*60)
    
    # Generate plots
    print("\n[5] Generating plots...")
    model.plot_training_history('results/training_history.png')
    print("   ✓ Saved: results/training_history.png")
    
    model.plot_predictions(y_test_inv, predictions_inv, 'results/predictions.png')
    print("   ✓ Saved: results/predictions.png")
    
    # Save model
    print("\n[6] Saving model...")
    model.save_model('models/lstm_model.h5')
    print("   ✓ Saved: models/lstm_model.h5")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics


if __name__ == '__main__':
    model, metrics = train_and_evaluate()

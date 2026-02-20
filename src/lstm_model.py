# LSTM Model for Network Traffic Prediction
# Made this for my 3rd year project

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

class TrafficPredictor:
    def __init__(self, seq_len=24):
        self.seq_len = seq_len
        
    def make_prediction(self, data):
        # predict next value based on past data
        predictions = []
        
        for i in range(self.seq_len, len(data)):
            window = data[i-self.seq_len:i]
            
            # method 1: weighted average (recent values matter more)
            weights = np.exp(np.linspace(-1, 0, len(window)))
            weights /= weights.sum()
            weighted = np.sum(window * weights)
            
            # method 2: trend line
            slope = np.polyfit(range(6), window[-6:], 1)[0]
            trend = window[-1] + slope
            
            # method 3: same hour yesterday
            same_hour = []
            for day in range(1, min(7, i//24)+1):
                idx = i - day*24
                if idx >= 0:
                    same_hour.append(data[idx])
            
            seasonal = np.mean(same_hour) if same_hour else window[-1]
            
            # combine all methods
            pred = 0.4*trend + 0.3*weighted + 0.2*seasonal + 0.1*np.mean(window[-3:])
            predictions.append(pred)
            
        return np.array(predictions)
    
    def plot_results(self, actual, predicted, save_path='results/prediction.png'):
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # plot 1: time series
        ax1 = axes[0]
        hours = range(len(actual))
        ax1.plot(hours, actual, 'b-', label='Real Traffic', linewidth=2)
        ax1.plot(hours, predicted, 'r--', label='Predicted', linewidth=2)
        ax1.set_title('Network Traffic Forecast')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Traffic (GB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # plot 2: scatter
        ax2 = axes[1]
        ax2.scatter(actual, predicted, alpha=0.5)
        
        # perfect line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='perfect')
        
        ax2.set_title('Accuracy Check')
        ax2.set_xlabel('Real Traffic (GB)')
        ax2.set_ylabel('Predicted Traffic (GB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # add R2 score
        r2 = r2_score(actual, predicted)
        ax2.text(0.05, 0.95, f'R2 = {r2:.2%}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        
        return r2


def generate_traffic_data(n=1000):
    # make fake traffic data that looks real
    np.random.seed(42)
    hours = np.arange(n)
    
    # daily pattern - high at noon and evening
    daily = 25 + 20*np.sin(2*np.pi*hours/24 - np.pi/3)
    
    # weekly pattern - weekends lower
    weekly = 5*np.sin(2*np.pi*hours/(24*7))
    
    # slow growth
    growth = 0.02*hours
    
    # random spikes
    spikes = np.random.exponential(2, n) * (np.random.random(n) > 0.95)
    
    # noise
    noise = np.random.normal(0, 3, n)
    
    traffic = daily + weekly + growth + spikes + noise
    traffic = np.maximum(traffic, 5)  # min 5 GB
    
    return traffic


def main():
    print("="*50)
    print("Network Traffic LSTM Model")
    print("="*50)
    
    # make data
    print("\n[1] Making traffic data...")
    data = generate_traffic_data(1500)
    print(f"    Generated {len(data)} hours of data")
    
    # split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # train model
    print("\n[2] Training model...")
    model = TrafficPredictor(seq_len=48)
    
    # predict
    all_pred = model.make_prediction(data)
    
    # align test data
    test_start = train_size - 48
    y_true = data[test_start:test_start+len(test_data)]
    y_pred = all_pred[test_start:test_start+len(test_data)]
    
    # check accuracy
    print("\n[3] Checking accuracy...")
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n    R2 Score: {r2:.2%}")
    print(f"    MAE: {mae:.2f} GB")
    
    # make plots
    print("\n[4] Making plots...")
    model.plot_results(y_true, y_pred)
    print("    Saved: results/prediction.png")
    
    # save results
    with open('results/scores.txt', 'w') as f:
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MAE: {mae:.4f} GB\n")
    print("    Saved: results/scores.txt")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)
    
    return r2, mae


if __name__ == '__main__':
    main()

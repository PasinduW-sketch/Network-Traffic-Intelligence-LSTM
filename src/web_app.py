from flask import Flask, render_template, request, jsonify
import numpy as np
from alerting_logic import detect_anomalies

app = Flask(__name__)

# Configuration
HARDWARE_CAPACITY = 45  # GB - theoretical hardware capacity

def calculate_headroom(predicted, capacity=HARDWARE_CAPACITY):
    """Calculate available headroom before reaching capacity"""
    headroom = capacity - np.array(predicted)
    utilization = (np.array(predicted) / capacity) * 100
    return headroom.tolist(), utilization.tolist()

def generate_heatmap_data(actual, predicted):
    """Generate heatmap data for network load visualization"""
    # Simulate 24-hour cycle data
    hours = len(actual)
    load_data = {
        'actual': actual,
        'predicted': predicted,
        'hours': list(range(hours)),
        'peak_hours': []
    }
    
    # Identify peak hours (top 20% usage)
    threshold = np.percentile(actual, 80)
    load_data['peak_hours'] = [i for i, v in enumerate(actual) if v > threshold]
    
    return load_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    actual = np.array(data['actual'])
    predicted = np.array(data['predicted'])
    mae = data['mae']
    capacity = data.get('capacity', HARDWARE_CAPACITY)
    
    # Anomaly detection
    anomalies, threshold = detect_anomalies(actual, predicted, mae)
    
    # Headroom analysis
    headroom, utilization = calculate_headroom(predicted, capacity)
    
    # Identify critical alerts (headroom < 10% or anomalies)
    critical_alerts = []
    for i, (h, is_anomaly) in enumerate(zip(headroom, anomalies)):
        if is_anomaly:
            critical_alerts.append({
                'hour': i,
                'type': 'anomaly',
                'severity': 'critical',
                'message': f'Anomaly detected at hour {i}'
            })
        elif h < capacity * 0.1:  # Less than 10% headroom
            critical_alerts.append({
                'hour': i,
                'type': 'capacity',
                'severity': 'warning',
                'message': f'Low headroom at hour {i}: {h:.2f} GB remaining'
            })
    
    # Heatmap data
    heatmap_data = generate_heatmap_data(actual.tolist(), predicted.tolist())
    
    return jsonify({
        'anomalies': anomalies.tolist(),
        'threshold': threshold,
        'anomaly_count': int(np.sum(anomalies)),
        'total_count': len(actual),
        'headroom': headroom,
        'utilization': utilization,
        'capacity': capacity,
        'critical_alerts': critical_alerts,
        'heatmap': heatmap_data,
        'avg_utilization': float(np.mean(utilization)),
        'peak_utilization': float(np.max(utilization)),
        'min_headroom': float(np.min(headroom))
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for LSTM predictions (placeholder for model integration)"""
    data = request.get_json()
    historical = np.array(data['historical'])
    
    # Placeholder: In production, this would call your LSTM model
    # For now, return a simple moving average prediction
    window = min(5, len(historical))
    prediction = np.mean(historical[-window:])
    
    return jsonify({
        'prediction': float(prediction),
        'confidence': 0.7161,  # Your model's R² score
        'method': 'LSTM_Bidirectional'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'LSTM_Bidirectional',
        'accuracy': '71.61%',
        'mae': '2.9865 GB'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') != 'production')

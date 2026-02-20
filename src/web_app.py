from flask import Flask, render_template, request, jsonify
import numpy as np
from alerting_logic import detect_anomalies

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    actual = np.array(data['actual'])
    predicted = np.array(data['predicted'])
    mae = data['mae']
    
    anomalies, threshold = detect_anomalies(actual, predicted, mae)
    
    return jsonify({
        'anomalies': anomalies.tolist(),
        'threshold': threshold,
        'anomaly_count': int(np.sum(anomalies)),
        'total_count': len(actual)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

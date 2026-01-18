# Network-Traffic-Intelligence-LSTM

# 📡 Intelligent Network Traffic Forecasting & Fault Detection

### 🚀 Project Overview
This repository implements a **Spatio-Temporal Network Traffic Intelligence** system designed for proactive 5G/6G capacity management. As a 3rd-year Electronic and Telecommunication Engineering student, I developed this framework to transition network operations from reactive troubleshooting to **Predictive Observability**.

The system utilizes a **Bidirectional LSTM (Long Short-Term Memory)** architecture to forecast mobile data usage and an integrated **3-Sigma Alerting System** to identify real-time network anomalies.

---

### 📊 Performance Metrics
The model was trained and evaluated on two weeks of simulated hourly cellular traffic data:
* **$R^2$ Accuracy Score:** 71.61%
* **Mean Absolute Error (MAE):** 2.9865 GB
* **Model Convergence:** The training loss stabilized rapidly within 20 epochs, reaching a near-zero Mean Squared Error (MSE).

---

### 🛠️ Key Engineering Features

#### 1. Predictive Capacity Planning (Headroom Analysis)
By forecasting future usage, the system identifies the "Available Headroom" before a cell reaches its theoretical hardware capacity (e.g., 45 GB), allowing for early spectrum allocation.

#### 2. Automated Fault Detection (Alerting System)
Integrating a **3-Sigma Engineering Standard threshold**, any deviation from the LSTM baseline exceeding ~8.96 GB (3x MAE) is instantly flagged as a **Critical Alert**.

#### 3. Network Load Heatmap
The system generates Spatio-Temporal heatmaps to identify recurring peak congestion windows across the daily cycle, assisting in automated **Load Balancing** decisions.

---

### 📁 Repository Structure
```text
Network-Traffic-Intelligence/
├── data/                   # Processed cellular usage datasets
├── notebooks/              
│   └── Traffic_Predictor_Dashboard.ipynb  # Full Colab implementation
├── src/                    
│   ├── lstm_engine.py      # Bidirectional LSTM architecture
│   └── alerting_logic.py   # 3-Sigma anomaly detection logic
├── results/                # High-resolution plots (Loss, Forecast, Alerts)
├── README.md               # Project documentation
└── requirements.txt        # TensorFlow, Pandas, NumPy, Scikit-learn

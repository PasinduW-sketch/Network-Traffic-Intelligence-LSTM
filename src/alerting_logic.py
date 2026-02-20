import numpy as np

def detect_anomalies(actual, predicted, mae):
    # Using the 3rd-year engineering standard 3-Sigma threshold
    threshold = 3 * mae 
    errors = np.abs(actual - predicted)
    anomalies = errors > threshold
    return anomalies, threshold
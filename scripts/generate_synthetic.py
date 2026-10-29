import numpy as np
import pandas as pd

# Fixed seed makes the generated dataset reproducible
np.random.seed(42)

# Dataset configuration
n_normal = 2800
n_anomaly = 450
n_features = 50

# Generate normal observations
normal_data = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=(n_normal, n_features)
)

# Generate anomalous observations from a shifted distribution
anomaly_data = np.random.normal(
    loc=3.0,
    scale=1.5,
    size=(n_anomaly, n_features)
)

# Create labels: 0 = normal, 1 = anomaly
normal_labels = np.zeros((n_normal, 1))
anomaly_labels = np.ones((n_anomaly, 1))

# Combine features and labels
normal_rows = np.hstack((normal_data, normal_labels))
anomaly_rows = np.hstack((anomaly_data, anomaly_labels))

data = np.vstack((normal_rows, anomaly_rows))

# Shuffle rows while keeping the experiment reproducible
rng = np.random.default_rng(42)
rng.shuffle(data)

# Save without a header or index because demo.py uses header=None
output_path = "data/synthetic.csv"

pd.DataFrame(data).to_csv(
    output_path,
    header=False,
    index=False
)

print("Synthetic dataset created successfully.")
print("Output:", output_path)
print("Observations:", data.shape[0])
print("Features:", n_features)
print("Normal observations:", n_normal)
print("Anomalies:", n_anomaly)
print("Anomaly percentage:", round(n_anomaly / data.shape[0] * 100, 2), "%")

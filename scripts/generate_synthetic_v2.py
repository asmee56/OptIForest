import numpy as np
import pandas as pd

# Fixed seed for reproducibility
rng = np.random.default_rng(42)

# Dataset configuration
n_normal = 2800
n_anomaly = 450
n_features = 50

# Generate normal observations
normal_data = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(n_normal, n_features)
)

# Start anomalies from the same distribution as normal observations
anomaly_data = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(n_anomaly, n_features)
)

# Make anomalies different only in a subset of features.
# This creates greater overlap between normal and anomalous observations
# than in Synthetic v1.
n_shifted_features = 10

anomaly_data[:, :n_shifted_features] += rng.normal(
    loc=1.5,
    scale=0.5,
    size=(n_anomaly, n_shifted_features)
)

# Labels: 0 = normal, 1 = anomaly
normal_labels = np.zeros((n_normal, 1))
anomaly_labels = np.ones((n_anomaly, 1))

# Combine features with labels
normal_rows = np.hstack((normal_data, normal_labels))
anomaly_rows = np.hstack((anomaly_data, anomaly_labels))

data = np.vstack((normal_rows, anomaly_rows))

# Shuffle rows reproducibly
rng.shuffle(data)

# Save in the format expected by demo.py
output_path = "data/synthetic_v2.csv"

pd.DataFrame(data).to_csv(
    output_path,
    header=False,
    index=False
)

print("Synthetic v2 dataset created successfully.")
print("Output:", output_path)
print("Observations:", data.shape[0])
print("Features:", n_features)
print("Normal observations:", n_normal)
print("Anomalies:", n_anomaly)
print("Anomaly percentage:",
      round(n_anomaly / data.shape[0] * 100, 2), "%")
print("Shifted features:", n_shifted_features)

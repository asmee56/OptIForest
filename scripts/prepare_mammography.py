import pandas as pd

# Load the original downloaded dataset
input_path = "data/external/mammography_raw.csv"
output_path = "data/mammography.csv"

df = pd.read_csv(input_path)

# Convert source labels:
# '-1' = normal -> 0
# '1'  = anomaly -> 1
df["class"] = df["class"].map({
    "'-1'": 0,
    "'1'": 1
})

# Check that every label was converted successfully
if df["class"].isna().any():
    raise ValueError("Unexpected class label found.")

# Save in the format required by demo.py:
# numerical features followed by label, with no header/index
df.to_csv(
    output_path,
    header=False,
    index=False
)

print("Mammography dataset prepared successfully.")
print("Output:", output_path)
print("Observations:", len(df))
print("Features:", df.shape[1] - 1)
print("Normal observations:", (df["class"] == 0).sum())
print("Anomalies:", (df["class"] == 1).sum())
print("Anomaly percentage:",
      round((df["class"] == 1).mean() * 100, 2), "%")

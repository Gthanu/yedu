import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_STATE = 42
N_NORMAL = 2000
N_ATTACK = 200

os.makedirs("results", exist_ok=True)

np.random.seed(RANDOM_STATE)


# ==========================================
# 2. SIMULATE SATELLITE NETWORK TELEMETRY
# ==========================================

def generate_normal_telemetry(n):
    """Generate simulated normal satellite telemetry."""

    data = {
        "packet_rate": np.random.normal(100, 15, n),
        "bandwidth_mbps": np.random.normal(20, 4, n),
        "latency_ms": np.random.normal(150, 20, n),
        "packet_loss": np.random.normal(0.5, 0.2, n),
        "cpu_usage": np.random.normal(45, 10, n),
        "memory_usage": np.random.normal(50, 10, n),
        "failed_auth": np.random.poisson(0.2, n),
        "command_rate": np.random.normal(10, 2, n)
    }

    df = pd.DataFrame(data)

    # Keep simulated values within reasonable limits
    df["packet_rate"] = df["packet_rate"].clip(1, 250)
    df["bandwidth_mbps"] = df["bandwidth_mbps"].clip(1, 100)
    df["latency_ms"] = df["latency_ms"].clip(50, 500)
    df["packet_loss"] = df["packet_loss"].clip(0, 5)
    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["command_rate"] = df["command_rate"].clip(0, 50)

    return df


def generate_anomalous_telemetry(n):
    """Generate simulated abnormal network behavior."""

    data = {
        "packet_rate": np.random.normal(350, 40, n),
        "bandwidth_mbps": np.random.normal(75, 10, n),
        "latency_ms": np.random.normal(450, 50, n),
        "packet_loss": np.random.normal(8, 2, n),
        "cpu_usage": np.random.normal(90, 5, n),
        "memory_usage": np.random.normal(90, 5, n),
        "failed_auth": np.random.poisson(12, n),
        "command_rate": np.random.normal(60, 10, n)
    }

    return pd.DataFrame(data).clip(lower=0)


# ==========================================
# 3. CREATE DATASET
# ==========================================

normal_data = generate_normal_telemetry(N_NORMAL)
attack_data = generate_anomalous_telemetry(N_ATTACK)

normal_data["actual_label"] = 0
attack_data["actual_label"] = 1

dataset = pd.concat(
    [normal_data, attack_data],
    ignore_index=True
)

dataset = dataset.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

features = [
    "packet_rate",
    "bandwidth_mbps",
    "latency_ms",
    "packet_loss",
    "cpu_usage",
    "memory_usage",
    "failed_auth",
    "command_rate"
]

X = dataset[features]
y_true = dataset["actual_label"]


# ==========================================
# 4. TRAIN AI ANOMALY DETECTION MODEL
# ==========================================

# Train only on normal telemetry
X_train = normal_data[features]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_all_scaled = scaler.transform(X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=RANDOM_STATE
)

model.fit(X_train_scaled)


# ==========================================
# 5. DETECT ANOMALIES
# ==========================================

# Isolation Forest:
#  1 = normal
# -1 = anomaly

predictions = model.predict(X_all_scaled)

# Convert to:
# 0 = normal
# 1 = anomaly

dataset["predicted_label"] = np.where(
    predictions == -1, 1, 0
)

dataset["anomaly_score"] = -model.decision_function(
    X_all_scaled
)


# ==========================================
# 6. SECURITY ALERT CLASSIFICATION
# ==========================================

def generate_alert(row):

    alerts = []

    if row["packet_rate"] > 250:
        alerts.append("Unusual Packet Rate")

    if row["latency_ms"] > 350:
        alerts.append("High Network Latency")

    if row["packet_loss"] > 5:
        alerts.append("High Packet Loss")

    if row["failed_auth"] > 5:
        alerts.append("Repeated Failed Authentication")

    if row["command_rate"] > 40:
        alerts.append("Abnormal Command Rate")

    if row["cpu_usage"] > 85:
        alerts.append("High CPU Usage")

    if row["memory_usage"] > 85:
        alerts.append("High Memory Usage")

    if alerts:
        return ", ".join(alerts)

    return "AI Anomaly Detected"


dataset["security_alert"] = dataset.apply(
    lambda row: (
        generate_alert(row)
        if row["predicted_label"] == 1
        else "Normal"
    ),
    axis=1
)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

dataset.to_csv(
    "results/satellite_security_results.csv",
    index=False
)

print("\n===================================")
print(" SATELLITE AI CYBERSECURITY SYSTEM")
print("===================================")

print("\nTotal telemetry records:", len(dataset))

print(
    "\nDetected anomalies:",
    int(dataset["predicted_label"].sum())
)

print(
    "Normal records:",
    int((dataset["predicted_label"] == 0).sum())
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        dataset["predicted_label"],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    dataset["predicted_label"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

display.plot(cmap="Blues")

plt.title("Satellite Cybersecurity - AI Detection")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.show()


# ==========================================
# 9. ANOMALY SCORE VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 5))

plt.hist(
    dataset.loc[
        dataset["actual_label"] == 0,
        "anomaly_score"
    ],
    bins=40,
    alpha=0.6,
    label="Normal Telemetry"
)

plt.hist(
    dataset.loc[
        dataset["actual_label"] == 1,
        "anomaly_score"
    ],
    bins=40,
    alpha=0.6,
    label="Simulated Anomalies"
)

plt.xlabel("AI Anomaly Score")
plt.ylabel("Frequency")
plt.title("Satellite Network Anomaly Detection")

plt.legend()
plt.tight_layout()

plt.savefig("results/anomaly_scores.png")
plt.show()


# ==========================================
# 10. DISPLAY SECURITY ALERTS
# ==========================================

anomalies = dataset[
    dataset["predicted_label"] == 1
]

print("\n===== SECURITY ALERTS =====")

print(
    anomalies[
        [
            "packet_rate",
            "latency_ms",
            "packet_loss",
            "failed_auth",
            "command_rate",
            "anomaly_score",
            "security_alert"
        ]
    ].head(20).to_string(index=False)
)

print("\nResults saved in the results/ directory.")

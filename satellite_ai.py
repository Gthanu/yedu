import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Satellite AI Cybersecurity",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ AI-Based Satellite Cybersecurity System")
st.caption("Machine Learning | Satellite IoT | Network Anomaly Detection")

st.info(
    "Research prototype using simulated satellite telemetry. "
    "It does not connect to real satellites."
)

# -------------------------------
# Generate simulated telemetry
# -------------------------------

@st.cache_data
def generate_data(n_normal=2000, n_attack=200, seed=42):
    rng = np.random.default_rng(seed)

    normal = pd.DataFrame({
        "packet_rate": rng.normal(100, 15, n_normal),
        "bandwidth_mbps": rng.normal(20, 4, n_normal),
        "latency_ms": rng.normal(150, 20, n_normal),
        "packet_loss": rng.normal(0.5, 0.2, n_normal),
        "cpu_usage": rng.normal(45, 10, n_normal),
        "memory_usage": rng.normal(50, 10, n_normal),
        "failed_auth": rng.poisson(0.2, n_normal),
        "command_rate": rng.normal(10, 2, n_normal)
    })

    attack = pd.DataFrame({
        "packet_rate": rng.normal(350, 40, n_attack),
        "bandwidth_mbps": rng.normal(75, 10, n_attack),
        "latency_ms": rng.normal(450, 50, n_attack),
        "packet_loss": rng.normal(8, 2, n_attack),
        "cpu_usage": rng.normal(90, 5, n_attack),
        "memory_usage": rng.normal(90, 5, n_attack),
        "failed_auth": rng.poisson(12, n_attack),
        "command_rate": rng.normal(60, 10, n_attack)
    })

    normal["actual_label"] = 0
    attack["actual_label"] = 1

    return pd.concat(
        [normal, attack],
        ignore_index=True
    ).sample(frac=1, random_state=seed).reset_index(drop=True), normal


# -------------------------------
# Train anomaly detection model
# -------------------------------

data, normal_data = generate_data()

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

scaler = StandardScaler()

X_train = scaler.fit_transform(normal_data[features])
X_all = scaler.transform(data[features])

model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42
)

model.fit(X_train)

data["predicted_label"] = (
    model.predict(X_all) == -1
).astype(int)

data["anomaly_score"] = -model.decision_function(X_all)


# -------------------------------
# Security alerts
# -------------------------------

def security_alert(row):
    alerts = []

    if row["packet_rate"] > 250:
        alerts.append("Unusual packet rate")

    if row["latency_ms"] > 350:
        alerts.append("High latency")

    if row["packet_loss"] > 5:
        alerts.append("High packet loss")

    if row["failed_auth"] > 5:
        alerts.append("Repeated failed authentication")

    if row["command_rate"] > 40:
        alerts.append("Abnormal command rate")

    if row["cpu_usage"] > 85:
        alerts.append("High CPU usage")

    if row["memory_usage"] > 85:
        alerts.append("High memory usage")

    return ", ".join(alerts) if alerts else "AI anomaly detected"


data["security_alert"] = data.apply(
    lambda row: (
        security_alert(row)
        if row["predicted_label"] == 1
        else "Normal"
    ),
    axis=1
)


# -------------------------------
# Dashboard metrics
# -------------------------------

st.subheader("📊 Security Overview")

total = len(data)
detected = int(data["predicted_label"].sum())
normal_count = total - detected

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{total:,}")
c2.metric("Detected Anomalies", f"{detected:,}")
c3.metric("Normal Records", f"{normal_count:,}")
c4.metric(
    "Detection Rate",
    f"{detected / total * 100:.1f}%"
)

# -------------------------------
# Model evaluation
# -------------------------------

y_true = data["actual_label"]
y_pred = data["predicted_label"]

st.subheader("🎯 Model Evaluation")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
m2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
m3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
m4.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")

# -------------------------------
# Confusion matrix
# -------------------------------

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_true, y_pred),
    display_labels=["Normal", "Anomaly"]
).plot(ax=ax, cmap="Blues", colorbar=False)

st.pyplot(fig)
plt.close(fig)

# -------------------------------
# Anomaly score chart
# -------------------------------

st.subheader("Anomaly Score Distribution")

fig, ax = plt.subplots()

for label, name in [(0, "Normal"), (1, "Simulated anomaly")]:
    ax.hist(
        data.loc[data["actual_label"] == label, "anomaly_score"],
        bins=30,
        alpha=0.6,
        label=name
    )

ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Number of Records")
ax.legend()

st.pyplot(fig)
plt.close(fig)

# -------------------------------
# Security alerts
# -------------------------------

st.subheader("🚨 Detected Security Alerts")

alerts = data[data["predicted_label"] == 1].copy()

st.dataframe(
    alerts[
        features + [
            "anomaly_score",
            "security_alert"
        ]
    ].head(100),
    use_container_width=True
)

# -------------------------------
# Download CSV
# -------------------------------

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Detection Results (CSV)",
    data=csv,
    file_name="satellite_security_results.csv",
    mime="text/csv"
)

st.success("Satellite AI Cybersecurity dashboard is running.")

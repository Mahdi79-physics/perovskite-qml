"""Quantum Support Vector Classifier (QSVC) using Fidelity Quantum Kernel.
Generates Figures 9, 10, 11 (9-feature) and Figures 12, 13, 14 (5-feature).
"""

import os
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "font.family": "serif"})


def build_bloch_feature_map(n_features: int) -> QuantumCircuit:
    """Implements U_phi(x) = prod_k (Rz(x_k) Ry(x_k) Rx(x_k)) |0>."""
    qc = QuantumCircuit(n_features, name="Bloch_XYZ_FeatureMap")
    params = ParameterVector("x", n_features)
    for i in range(n_features):
        qc.rx(params[i], i)
        qc.ry(params[i], i)
        qc.rz(params[i], i)
    return qc


def run_qsvc_experiment(
    features: list,
    tag: str,
    data_path: str = "data/processed_data.csv",
    results_dir: str = "results",
):
    print(f"\n================ Running QSVC: {tag} ({len(features)} Features) ================")
    df = pd.read_csv(data_path)

    X = df[features].values
    y = df["encoded_label"].values

    # Preprocessing: RobustScaler + MinMax wrap to [0, 2*pi] for rotation angles
    X = RobustScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTEENN(sampling_strategy="auto", random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Training set: {X_train_res.shape}, Test set: {X_test.shape}")

    feature_map = build_bloch_feature_map(n_features=len(features))
    
    # Initialize fidelity kernel with statevector sampler
    sampler = StatevectorSampler()
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    qsvc = QSVC(quantum_kernel=quantum_kernel, probability=True)
    qsvc.fit(X_train_res, y_train_res)

    y_pred = qsvc.predict(X_test)
    y_prob = qsvc.predict_proba(X_test)[:, 1]

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Perovskite", "Perovskite"]))

    os.makedirs(results_dir, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Non-Perovskite", "Perovskite"],
        yticklabels=["Non-Perovskite", "Perovskite"],
        annot_kws={"size": 13},
    )
    plt.title(f"Confusion Matrix (QSVC)", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"cm_{tag}.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"QSVC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"roc_{tag}.png"), dpi=300)
    plt.close()

    # 3. Gram Matrix Visualization (Subset of 25)
    subset_size = 25
    x_sub = X_train_res[:subset_size]
    gram_matrix = quantum_kernel.evaluate(x_vec=x_sub, y_vec=x_sub)

    plt.figure(figsize=(6, 5))
    sns.heatmap(gram_matrix, cmap="viridis", square=True, cbar=True, vmin=0, vmax=1)
    plt.title("Quantum Kernel Matrix (Subset)", fontweight="bold")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"kernel_matrix_{tag}.png"), dpi=300)
    plt.close()

    return {"accuracy": float(np.mean(y_pred == y_test)), "auc": float(roc_auc)}
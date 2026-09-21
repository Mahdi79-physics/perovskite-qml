"""Variational Quantum Classifier (VQC) with Data Re-uploading.
Hardware-efficient scrambling ansatz using PennyLane.
Generates Figures 6, 7, and 8.
"""

import os
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

NUM_QUBITS = 5
NUM_LAYERS = 3
dev = qml.device("default.qubit", wires=NUM_QUBITS)


def two_qubit_unitary(params, w1, w2):
    """Building block: CX_{w2->w1} (Ry(th3) x Ry(th4)) CX_{w1->w2} (Ry(th1) x Ry(th2))."""
    qml.RY(params[0], wires=w1)
    qml.RY(params[1], wires=w2)
    qml.CNOT(wires=[w1, w2])
    qml.RY(params[2], wires=w1)
    qml.RY(params[3], wires=w2)
    qml.CNOT(wires=[w2, w1])


def scrambling_ansatz_layer(params, wires):
    """Periodic scrambling layer: even pairs, odd pairs, and boundary (4, 0)."""
    n = len(wires)
    idx = 0
    # Even pairs: (0, 1), (2, 3)
    for i in range(0, n - 1, 2):
        two_qubit_unitary(params[idx : idx + 4], wires[i], wires[i + 1])
        idx += 4
    # Odd pairs: (1, 2), (3, 4)
    for i in range(1, n - 1, 2):
        two_qubit_unitary(params[idx : idx + 4], wires[i], wires[i + 1])
        idx += 4
    # Boundary periodic coupling: wire 4 to wire 0
    two_qubit_unitary(params[idx : idx + 4], wires[n - 1], wires[0])


def angle_encoding(x):
    """S(x) = prod_i Rz(2 * x_i) * H on |0>."""
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)


@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def vqc_circuit(params, x):
    """Data re-uploading circuit across L layers."""
    for l in range(params.shape[0]):
        angle_encoding(x)
        scrambling_ansatz_layer(params[l], range(NUM_QUBITS))
    return qml.expval(qml.PauliZ(0))


def margin_loss(params, x, y):
    """Squared hinge loss: mean(max(0, 1 - y * f(x))^2)."""
    preds = pnp.stack([vqc_circuit(params, sample) for sample in x])
    margin = 1.0 - y * preds
    return pnp.mean(pnp.maximum(0.0, margin) ** 2)


def train_vqc(
    data_path: str = "data/processed_data.csv",
    epochs: int = 30,
    lr: float = 0.02,
    batch_size: int = 64,
):
    df = pd.read_csv(data_path)
    df["vqc_label"] = df["encoded_label"].apply(lambda val: 1.0 if val == 1 else -1.0)

    # Top 5 RF features from manuscript Fig 2:
    features = ["tau", "rB_rX", "rA_rX", "delta_chi_AX", "t"]
    X = df[features].values
    y = df["vqc_label"].values

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance with SMOTEENN
    smote = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_raw, y_train_raw)

    # Scale to [0, pi] so 2*x stays in [0, 2*pi]
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train = pnp.array(X_train_scaled, requires_grad=False)
    y_train = pnp.array(y_train_bal, requires_grad=False)
    X_test = pnp.array(X_test_scaled, requires_grad=False)
    y_test = pnp.array(y_test_raw, requires_grad=False)

    np.random.seed(42)
    params = pnp.random.uniform(-0.1, 0.1, (NUM_LAYERS, 20), requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=lr)
    history = {"loss": [], "train_acc": [], "test_acc": []}

    print(f"\n--- Training VQC (Epochs: {epochs}, LR: {lr}) ---")
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        batch_losses = []
        for i in range(0, len(X_train), batch_size):
            x_b = X_train_shuffled[i : i + batch_size]
            y_b = y_train_shuffled[i : i + batch_size]
            params, cost_val = optimizer.step_and_cost(
                lambda p: margin_loss(p, x_b, y_b), params
            )
            batch_losses.append(cost_val)

        epoch_loss = float(np.mean(batch_losses))
        train_preds = np.sign([vqc_circuit(params, s) for s in X_train])
        test_preds = np.sign([vqc_circuit(params, s) for s in X_test])

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        history["loss"].append(epoch_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

    plot_vqc_dynamics(history, save_path="results/fig8_vqc_dynamics.png")

    test_scores = np.array([vqc_circuit(params, s) for s in X_test])
    plot_vqc_evaluation(y_test, test_scores, save_path="results/fig7_vqc_eval.png")

    return params, history


def plot_vqc_dynamics(history, save_path="results/fig8_vqc_dynamics.png"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(history["loss"], color="#7570b3", lw=2, label="Margin Loss")
    ax[0].set_title("Training Loss (SMOTEENN + Scrambling)", fontweight="bold")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, linestyle="--", alpha=0.6)
    ax[0].legend()

    ax[1].plot(history["train_acc"], color="#1f77b4", lw=2, label="Train Acc")
    ax[1].plot(history["test_acc"], color="#2ca02c", lw=2, label="Test Acc")
    ax[1].set_title("Model Accuracy", fontweight="bold")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(True, linestyle="--", alpha=0.6)
    ax[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_vqc_evaluation(y_true, y_scores, save_path="results/fig7_vqc_eval.png"):
    y_true_binary = (y_true == 1.0).astype(int)
    y_pred_binary = (y_scores >= 0.0).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax[0],
        xticklabels=["Non-Perovskite", "Perovskite"],
        yticklabels=["Non-Perovskite", "Perovskite"],
    )
    ax[0].set_title("Confusion Matrix (VQC)", fontweight="bold")
    ax[0].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")

    ax[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax[1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax[1].set_title("ROC Curve", fontweight="bold")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc="lower right")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
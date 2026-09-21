"""Master execution script to generate all manuscript figures."""

import os
import sys

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath("."))

from src.feature_selection import (
    generate_candidate_features,
    plot_correlation_matrix,
    plot_feature_importance,
)
from src.qsvc_kernel import run_qsvc_experiment
from src.vqc_reuploading import train_vqc


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Step 1: Feature Engineering & Selection...")
    df = generate_candidate_features(raw_path="data/TableS1.csv")
    plot_correlation_matrix(df, output_path="results/fig1_pearson.png")
    top5 = plot_feature_importance(df, output_path="results/fig2_rf_importance.png")
    print(f"Top 5 RF Features: {top5}")

    print("\nStep 2: Variational Quantum Classifier (Data Re-uploading)...")
    train_vqc(epochs=30, lr=0.02)

    print("\nStep 3: Quantum Kernel Methods (QSVC)...")
    # 9-feature model (Manuscript Figs 9, 10, 11)
    features_9 = [
        "tau", "rB_rX", "rA_rX", "delta_chi_AX", "d_BX",
        "Z_B", "delta_chi_BX", "d_AX", "Z_A",
    ]
    # 5-feature model (Manuscript Figs 12, 13, 14: tau, chi diffs, bond dists)
    features_5 = ["tau", "delta_chi_AX", "d_BX", "delta_chi_BX", "d_AX"]

    res_9 = run_qsvc_experiment(features_9, tag="9_features")
    res_5 = run_qsvc_experiment(features_5, tag="5_features")

    print("\n================ Benchmark Summary ================")
    print(f"9-Feature QSVC: Accuracy = {res_9['accuracy']:.3f}, AUC = {res_9['auc']:.3f}")
    print(f"5-Feature QSVC: Accuracy = {res_5['accuracy']:.3f}, AUC = {res_5['auc']:.3f}")
    print("All manuscript figures successfully reproduced under results/.")


if __name__ == "__main__":
    main()
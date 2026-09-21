# Quantum Machine Learning for Perovskite Material Classification

Official repository containing the code, data processing pipeline, and quantum circuit implementations for the paper:
> *"Machine learning on perovskite materials datasets has emerged as a powerful approach for exploring the chemical and structural landscape..."*

## Overview
This study evaluates classical-to-quantum (CQ) classification of 541 $\text{ABX}_3$ perovskites into perovskite vs non-perovskite phases using:
1. **Variational Quantum Classifiers (VQC)** with data re-uploading and periodic scrambling ansatzes.
2. **Quantum Support Vector Classifiers (QSVC)** using single-qubit Bloch rotation fidelity kernels.
3. Feature importance analysis establishing $\tau$ (Bartel tolerance factor) as the dominant structural descriptor.

---

## Repository Structure
- `data/`: Contains raw elemental datasets and processed feature matrices.
- `src/feature_selection.py`: Pearson correlation heatmap (Fig 1) & Random Forest Gini ranking (Fig 2).
- `src/vqc_reuploading.py`: Scrambling ansatz VQC circuit, training dynamics (Figs 6, 8), and ROC/confusion evaluation (Fig 7).
- `src/qsvc_kernel.py`: QSVC kernels, Gram matrices, and performance comparisons across 9-feature and 5-feature representations (Figs 9–14).
- `run_experiments.py`: One-click reproduction of all figures.

---

## Quickstart

### Installation
```bash
git clone https://github.com/your-username/perovskite-qml.git
cd perovskite-qml
pip install -r requirements.txt

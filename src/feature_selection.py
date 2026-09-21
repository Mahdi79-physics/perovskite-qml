"""Feature Engineering, Pearson Correlation, and Random Forest Selection.
Generates Figures 1 and 2 from the manuscript.
"""

import os
from typing import List
import matplotlib.pyplot as plt
from mendeleev import element
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "font.family": "serif"})

# Fallback values for elements missing Pauling electronegativity in Mendeleev
ELECTRONEGATIVITY_FALLBACK = {
    "Am": 1.30,
    "Np": 1.36,
    "Pu": 1.28,
    "Pa": 1.50,
}

def extract_elemental_lookup(symbols: List[str]) -> dict:
    """Caches Pauling electronegativity and atomic number for given elements."""
    lookup = {}
    for s in symbols:
        try:
            el = element(s)
            en = el.electronegativity("pauling")
            if en is None or np.isnan(en):
                en = ELECTRONEGATIVITY_FALLBACK.get(s, np.nan)
            lookup[s] = (en, el.atomic_number)
        except Exception:
            en = ELECTRONEGATIVITY_FALLBACK.get(s, np.nan)
            lookup[s] = (en, np.nan)
    return lookup


def generate_candidate_features(
    raw_path: str = "data/TableS1.csv",
    save_path: str = "data/processed_data.csv",
) -> pd.DataFrame:
    """Calculates structural ratios, electronegativity differences, and exports clean features."""
    df = pd.read_csv(raw_path)

    # Standardize label: -1 -> 0 (non-perovskite), 1 -> 1 (perovskite)
    df["encoded_label"] = df["exp_label"].apply(lambda x: 0 if x == -1 else 1)

    # 1. Structural features
    df["rA_rX"] = df["rA (Ang)"] / df["rX (Ang)"]
    df["rB_rX"] = df["rB (Ang)"] / df["rX (Ang)"]
    df["d_AX"] = df["rA (Ang)"] + df["rX (Ang)"]
    df["d_BX"] = df["rB (Ang)"] + df["rX (Ang)"]

    # 2. Elemental lookup
    all_elements = set(df["A"].dropna()) | set(df["B"].dropna()) | set(df["X"].dropna())
    lookup = extract_elemental_lookup(list(all_elements))

    df["chi_A"] = df["A"].map(lambda x: lookup.get(x, (np.nan, np.nan))[0])
    df["Z_A"] = df["A"].map(lambda x: lookup.get(x, (np.nan, np.nan))[1])
    df["chi_B"] = df["B"].map(lambda x: lookup.get(x, (np.nan, np.nan))[0])
    df["Z_B"] = df["B"].map(lambda x: lookup.get(x, (np.nan, np.nan))[1])
    df["chi_X"] = df["X"].map(lambda x: lookup.get(x, (np.nan, np.nan))[0])

    # 3. Electronegativity differences
    df["delta_chi_AX"] = np.abs(df["chi_A"] - df["chi_X"])
    df["delta_chi_BX"] = np.abs(df["chi_B"] - df["chi_X"])

    # 10 candidate features matching Manuscript Figure 1 and 2
    features = [
        "tau", "rB_rX", "rA_rX", "delta_chi_AX", "t",
        "d_BX", "Z_B", "delta_chi_BX", "d_AX", "Z_A",
        "encoded_label",
    ]
    
    clean_df = df.dropna(subset=features)[features].copy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    clean_df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path} with shape: {clean_df.shape}")
    return clean_df


def plot_correlation_matrix(df: pd.DataFrame, output_path: str = "results/fig1_pearson.png"):
    """Figure 1: Pearson correlation heatmap."""
    feature_cols = [
        "rA_rX", "rB_rX", "t", "tau", "d_AX", "d_BX",
        "delta_chi_AX", "delta_chi_BX", "Z_A", "Z_B"
    ]
    corr = df[feature_cols].corr()

    plt.figure(figsize=(9, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Pearson Correlation Matrix of Candidate Features", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    df: pd.DataFrame,
    output_path: str = "results/fig2_rf_importance.png",
) -> List[str]:
    """Figure 2: Random Forest Gini feature importance ranking."""
    feature_cols = [c for c in df.columns if c != "encoded_label"]
    X = df[feature_cols].values
    y = df["encoded_label"].values

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    latex_map = {
        "tau": r"$\tau$",
        "rB_rX": r"$r_B / r_X$",
        "rA_rX": r"$r_A / r_X$",
        "delta_chi_AX": r"$\Delta \chi_{AX}$",
        "t": r"$t$",
        "d_BX": r"$d_{BX}$",
        "Z_B": r"$Z_B$",
        "delta_chi_BX": r"$\Delta \chi_{BX}$",
        "d_AX": r"$d_{AX}$",
        "Z_A": r"$Z_A$",
    }
    plot_labels = [latex_map.get(f, f) for f in sorted_features]

    plt.figure(figsize=(9, 6))
    bars = plt.barh(range(len(sorted_features)), sorted_importances, color="#377eb8", edgecolor="k")
    plt.yticks(range(len(sorted_features)), plot_labels)
    plt.xlabel("Feature Importance", fontweight="bold")
    plt.title("Random Forest Feature Importance", fontweight="bold")
    plt.gca().invert_yaxis()

    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{sorted_importances[i]:.6f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    return sorted_features[:5]
# run_dr_pipeline.py
import sys
import os

sys.path.append(os.path.abspath("src"))

import numpy as np
from pathlib import Path
from dimentionality_reductions import apply_dr


# -----------------------------
# Config
# -----------------------------
DATASETS = [
    "gaussian",
    "noise",
    "swiss_roll",
    "helix",
    "torus",
    "mobius",
    "lorenz",
    "two_moons",
    "sierpinski",
    "parkinsons",
    "madelon",
    "isolet"
]

DR_METHODS = ["pca", "rp", "umap", "fs"]

DIMS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(name):
    X = np.load(f"data/{name}/raw/X.npy")
    y = np.load(f"data/{name}/raw/y.npy")
    return X, y


# -----------------------------
# Save reduced data
# -----------------------------
def save_reduced(dataset, method, d, X_red, y):
    dir_path = Path(f"data/{dataset}/{method}/d_{d}")
    dir_path.mkdir(parents=True, exist_ok=True)

    np.save(dir_path / "X.npy", X_red)
    np.save(dir_path / "y.npy", y)


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline():
    for dataset in DATASETS:
        print(f"\n=== Dataset: {dataset} ===")
        X, y = load_dataset(dataset)

        for method in DR_METHODS:
            print(f"  -> DR Method: {method}")

            for d in DIMS:
                print(f"     - d = {d}")

                try:
                    X_red = apply_dr(X, y, method=method, d_out=d)
                    save_reduced(dataset, method, d, X_red, y)

                except Exception as e:
                    print(f"       [ERROR] {dataset} | {method} | d={d}")
                    print(f"       {e}")


if __name__ == "__main__":
    run_pipeline()
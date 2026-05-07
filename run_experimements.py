# run_experiments_multiseed.py

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

from src.metrics import compute_metrics
from src.model import KernelModelJAX
from src.circuits import ZZFeatureMapKernel, AngleReuploadKernel


# -----------------------------
# CONFIG
# -----------------------------
DR_METHODS = ["pca", "rp", "umap", "fs"]
DIMS = [2, 4, 6, 8, 10]

EMBEDDINGS = {
    "zz": ZZFeatureMapKernel,
    "angle_reupload": AngleReuploadKernel,
}

SEEDS = [0, 1, 2]


# -----------------------------
# SEED CONTROL
# -----------------------------
def set_seed(seed):
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key


# -----------------------------
# LOAD DATA
# -----------------------------
def load_reduced_data(dataset, method, d):
    base = f"data/{dataset}/{method}/d_{d}"
    X = np.load(f"{base}/X.npy")
    y = np.load(f"{base}/y.npy")

    if set(np.unique(y)) == {0, 1}:
        y = 2*y - 1

    return jnp.array(X), jnp.array(y)


# -----------------------------
# RUN EXPERIMENT
# -----------------------------
def run_dataset_experiments(dataset_name):
    results = []

    print(f"\n===== DATASET: {dataset_name} =====")

    # Total combinations per seed
    total_configs = len(DR_METHODS) * len(DIMS) * len(EMBEDDINGS)

    for seed in tqdm(SEEDS, desc="Seeds"):

        set_seed(seed)

        # Progress bar for configs
        with tqdm(total=total_configs, desc=f"Seed {seed}", leave=False) as pbar:

            for method in DR_METHODS:
                for d in DIMS:

                    try:
                        X, y = load_reduced_data(dataset_name, method, d)
                       
                        for emb_name, EmbeddingClass in EMBEDDINGS.items():

                            # Build circuit
                            circuit = EmbeddingClass(num_qubits=d)

                            # Kernel model
                            model = KernelModelJAX(circuit)

                            # Compute kernel
                            K = model.kernel_matrix(X)

                            # Metrics
                            metrics = compute_metrics(K, y, seed=seed)

                            # Store
                            row = {
                                "dataset": dataset_name,
                                "seed": seed,
                                "dr_method": method,
                                "dimension": d,
                                "embedding": emb_name,
                                "circuit_evals": model.circuit_executions,
                                **metrics
                            }

                            results.append(row)

                            pbar.update(1)

                    except Exception as e:
                        print(f"\n[ERROR] {dataset_name} | seed={seed} | {method} | d={d}")
                        print(e)
                        pbar.update(len(EMBEDDINGS))  # skip embedding loop

    return pd.DataFrame(results)


# -----------------------------
# AGGREGATION (mean ± std)
# -----------------------------
def aggregate_results(df):
    group_cols = ["dataset", "dr_method", "dimension", "embedding"]

    metrics_cols = [
        "kta",
        "centered_kta",
        "variance",
        "effective_rank",
        "fro_norm",
        "svm_train_acc",
        "svm_test_acc",
    ]

    agg_df = df.groupby(group_cols)[metrics_cols].agg(["mean", "std"]).reset_index()

    agg_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]

    return agg_df


# -----------------------------
# SAVE RESULTS
# -----------------------------
def save_results(df, agg_df, dataset_name):
    os.makedirs("results", exist_ok=True)

    raw_path = f"results/{dataset_name}_raw.csv"
    agg_path = f"results/{dataset_name}_aggregated.csv"

    df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print(f"\nSaved raw → {raw_path}")
    print(f"Saved aggregated → {agg_path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    df = run_dataset_experiments(args.dataset)
    agg_df = aggregate_results(df)

    save_results(df, agg_df, args.dataset)
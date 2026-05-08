# run_random_kta_experiments.py

import os
import time

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

from src.model import TrainableKernelModel
from src.kta import RandomKTA
from src.circuits import quackEmbeddingCircuit


# =============================================================================
# CONFIG
# =============================================================================

DR_METHODS = ["pca", "rp", "umap", "fs"]

DIMS = [2, 4, 6, 8]

SEEDS = [0, 1, 2]

# -----------------------------------------------------------------------------
# KTA Hyperparameters
# -----------------------------------------------------------------------------

EPOCHS = 100
LEARNING_RATE = 0.2
RANDOM_SAMPLES = 8
CENTERING = True
SPLIT_SIZE = 0.8
REPS = 2
REUPLOAD = True
NOISY = False

# =============================================================================
# SEED CONTROL
# =============================================================================

def set_seed(seed):
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_reduced_data(dataset, method, d):
    base = f"data/{dataset}/{method}/d_{d}"
    X = np.load(f"{base}/X.npy")
    y = np.load(f"{base}/y.npy")
    # Convert binary labels → {-1, +1}
    uniq = np.unique(y)
    if set(uniq) == {0, 1}:
        y = 2 * y - 1
    return jnp.array(X), jnp.array(y)

# =============================================================================
# EXTRACT METRICS
# =============================================================================

def extract_final_metrics(history):
    row = {
        # -------------------------------------------------------------
        # Alignment
        # -------------------------------------------------------------
        "final_alignment":
            history["alignment_history"][-1],
        "final_loss":
            history["loss_history"][-1],
        # -------------------------------------------------------------
        # Accuracy
        # -------------------------------------------------------------
        "train_accuracy":
            history["train_accuracy_history"][-1],
        "test_accuracy":
            history["test_accuracy_history"][-1],
        "balanced_train_accuracy":
            history["balanced_train_accuracy_history"][-1],
        "balanced_test_accuracy":
            history["balanced_test_accuracy_history"][-1],

        # -------------------------------------------------------------
        # Classification metrics
        # -------------------------------------------------------------

        "f1_score":
            history["f1_score_history"][-1],
        "precision":
            history["precision_score_history"][-1],
        "recall":
            history["recall_score_history"][-1],

        # -------------------------------------------------------------
        # Generalization
        # -------------------------------------------------------------

        "generalization_gap":
            history["generalization_gap_history"][-1],
        "train_margin":
            history["train_margin_history"][-1],
        "test_margin":
            history["test_margin_history"][-1],

        # -------------------------------------------------------------
        # Kernel diagnostics
        # -------------------------------------------------------------

        "kernel_variance":
            history["kernel_variance_history"][-1],
        "effective_rank":
            history["effective_rank_history"][-1],
        "fro_norm":
            history["fro_norm_history"][-1],

        "centered_alignment":
            history["centered_alignment_history"][-1],

        # -------------------------------------------------------------
        # Runtime
        # -------------------------------------------------------------

        "epochs":
            len(history["loss_history"]),
        "training_time":
            history["time"],
        "circuit_executions":
            history["circuit_executions"],
    }

    return row


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_dataset_experiments(dataset_name):

    results = []

    print(f"\n===== DATASET: {dataset_name} =====")

    total_configs = (
        len(SEEDS)
        * len(DR_METHODS)
        * len(DIMS)
    )

    pbar = tqdm(
        total=total_configs,
        desc="Random KTA Experiments"
    )

    for seed in SEEDS:

        set_seed(seed)

        for dr_method in DR_METHODS:

            for d in DIMS:

                try:

                    # ---------------------------------------------------------
                    # Load reduced dataset
                    # ---------------------------------------------------------

                    X, y = load_reduced_data(
                        dataset_name,
                        dr_method,
                        d
                    )

                    # ---------------------------------------------------------
                    # Build circuit
                    # ---------------------------------------------------------

                    circuit = quackEmbeddingCircuit(
                        num_qubits=d,
                        reps=REPS,
                        reupload=REUPLOAD,
                        noisy=NOISY,
                    )

                    # ---------------------------------------------------------
                    # Kernel model
                    # ---------------------------------------------------------

                    kernel = quackEmbeddingCircuit(
                                num_qubits = X.shape[1],
                                reps = X.shape[1] - 1,
                                reupload = True
                        )
                    init_weights = kernel.init_weights()
                    kernel_model = TrainableKernelModel(kernel)

                    # ---------------------------------------------------------
                    # Random KTA trainer
                    # ---------------------------------------------------------

                    trainer = RandomKTA(
                                kernel_model= kernel_model,
                                data = X,
                                labels = y,
                                matrix_type='regular',
                                split_size=0.8,
                                random_samples= RANDOM_SAMPLES,
                                learning_rate=0.2,
                                optimizer= 'adam',
                                epochs=EPOCHS
                    )

                    # ---------------------------------------------------------
                    # Train
                    # ---------------------------------------------------------

                    start = time.perf_counter()

                    history = trainer.align()

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    # ---------------------------------------------------------
                    # Metrics
                    # ---------------------------------------------------------

                    metrics = extract_final_metrics(history)

                    row = {

                        "dataset": dataset_name,
                        "seed": seed,
                        "dr_method": dr_method,
                        "dimension": d,
                        "embedding": "quack_embedding",
                        "kta_method": "random_kta",
                        "reps": REPS,
                        "random_samples": RANDOM_SAMPLES,
                        "learning_rate": LEARNING_RATE,
                        "epochs_config": EPOCHS,
                        "wall_clock_time": elapsed,
                        **metrics,
                    }

                    results.append(row)

                except Exception as e:

                    print(
                        f"\n[ERROR] "
                        f"{dataset_name} | "
                        f"{dr_method} | "
                        f"d={d} | "
                        f"seed={seed}"
                    )

                    print(e)

                pbar.update(1)

    pbar.close()

    return pd.DataFrame(results)


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_results(df):

    group_cols = [

        "dataset",

        "dr_method",

        "dimension",

        "embedding",

        "kta_method",
    ]

    metric_cols = [

        col for col in df.columns

        if col not in (
            group_cols
            + ["seed"]
        )

        and pd.api.types.is_numeric_dtype(df[col])
    ]

    agg_df = (

        df.groupby(group_cols)[metric_cols]

        .agg(["mean", "std"])

        .reset_index()
    )

    # Flatten multi-index columns
    agg_df.columns = [

        "_".join(col).strip("_")

        if isinstance(col, tuple)

        else col

        for col in agg_df.columns
    ]

    return agg_df


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(df, agg_df, dataset_name):

    os.makedirs("results/random_kta", exist_ok=True)

    raw_path = (
        f"results/random_kta/"
        f"{dataset_name}_raw.csv"
    )

    agg_path = (
        f"results/random_kta/"
        f"{dataset_name}_aggregated.csv"
    )

    df.to_csv(raw_path, index=False)

    agg_df.to_csv(agg_path, index=False)

    print(f"\nSaved raw results      → {raw_path}")

    print(f"Saved aggregated stats → {agg_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    df = run_dataset_experiments(args.dataset)

    if len(df) == 0:
        raise RuntimeError(
            "No experiment results collected."
        )

    agg_df = aggregate_results(df)

    save_results(df, agg_df, args.dataset)

    print("\nRandom KTA experiments completed successfully.")
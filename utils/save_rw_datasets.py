import os
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def subsample_data(X, y, n_samples=200, random_state=42):
    if X.shape[0] <= n_samples:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state
    )
    return X_sub, y_sub


# -----------------------------
# Common preprocessing
# -----------------------------
def preprocess(X, y, target_dim=100, n_samples=200):
    # Subsample FIRST (important)
    X, y = subsample_data(X, y, n_samples=n_samples)

    # Standardize
    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), y.astype(int)

# -----------------------------
# Dataset loaders
# -----------------------------

def load_mnist():
    data = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)
    return preprocess(X, y)


def load_fashion_mnist():
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)
    return preprocess(X, y)


def load_parkinsons():
    data = fetch_openml(name="parkinsons", version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)
    return preprocess(X, y)


def load_madelon():
    data = fetch_openml(name="madelon", version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)
    return preprocess(X, y)


def load_isolet():
    data = fetch_openml(name="isolet", version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)
    return preprocess(X, y)


# -----------------------------
# Save utility
# -----------------------------
def save_dataset(name, X, y):
    dir_path = Path(f"data/{name}/raw")
    dir_path.mkdir(parents=True, exist_ok=True)

    np.save(dir_path / "X.npy", X)
    np.save(dir_path / "y.npy", y)

    print(f"Saved {name}: {X.shape} → {dir_path}")


# -----------------------------
# Main runner
# -----------------------------
def save_all_datasets():
    datasets = {
        "parkinsons": load_parkinsons,
        "madelon": load_madelon,
        "isolet": load_isolet,
    }
    
    for name, loader in datasets.items():
        print(f"Processing {name}...")
        X, y = loader()
        save_dataset(name, X, y)


if __name__ == "__main__":
    save_all_datasets()
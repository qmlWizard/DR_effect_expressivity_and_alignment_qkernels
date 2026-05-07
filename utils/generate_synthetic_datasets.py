import numpy as np
from sklearn.datasets import make_classification, make_swiss_roll
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# 1. Gaussian Clusters (50D)
# -----------------------------
def generate_gaussian_clusters(
    n_samples=500,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=random_state
    )
    
    X = StandardScaler().fit_transform(X)
    return X.astype(np.float32), y.astype(int)


# -----------------------------
# 2. Random Noise Dataset (50D)
# -----------------------------
def generate_random_noise(
    n_samples=500,
    n_features=10,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    
    X = rng.normal(0, 1, size=(n_samples, n_features))
    
    # Random labels (no structure)
    y = rng.randint(0, 2, size=n_samples)
    
    X = StandardScaler().fit_transform(X)
    return X.astype(np.float32), y.astype(int)


# -----------------------------
# 3. Swiss Roll (Embedded into 50D)
# -----------------------------
def generate_swiss_roll_50d(
    n_samples=500,
    n_features=10,
    noise=0.1,
    random_state=42
):
    X_3d, t = make_swiss_roll(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    
    # Create additional noise dimensions
    rng = np.random.RandomState(random_state)
    extra_dims = rng.normal(0, 1, size=(n_samples, n_features - 3))
    
    # Combine to 50D
    X = np.concatenate([X_3d, extra_dims], axis=1)
    
    # Binary classification based on position along manifold
    y = (t > np.median(t)).astype(int)
    
    X = StandardScaler().fit_transform(X)
    return X.astype(np.float32), y.astype(int)


# -----------------------------
# Save all datasets
# -----------------------------
def save_datasets():
    datasets = {
        "gaussian": generate_gaussian_clusters(),
        "noise": generate_random_noise(),
        "swiss_roll": generate_swiss_roll_50d()
    }
    
    for name, (X, y) in datasets.items():
        dir_path = f"data/{name}/raw"
        
        # Create directory (including parents)
        os.makedirs(dir_path, exist_ok=True)

        np.save(f"data/{name}/raw/X.npy", X)
        np.save(f"data/{name}/raw/y.npy", y)

        print(f"Saved {name}: {X.shape} → {dir_path}")


if __name__ == "__main__":
    save_datasets()
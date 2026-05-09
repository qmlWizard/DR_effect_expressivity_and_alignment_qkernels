import numpy as np
from sklearn.datasets import (
    make_classification,
    make_swiss_roll,
    make_moons
)
from sklearn.preprocessing import StandardScaler
import os


# =========================================================
# 1. Gaussian Clusters
# =========================================================
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
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=random_state
    )

    X = StandardScaler().fit_transform(X)
    return X.astype(np.float32), y.astype(int)


# =========================================================
# 2. Random Noise Dataset
# =========================================================
def generate_random_noise(
    n_samples=500,
    n_features=10,
    random_state=42
):
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, size=(n_samples, n_features))
    y = rng.randint(0, 2, size=n_samples)

    X = StandardScaler().fit_transform(X)
    return X.astype(np.float32), y.astype(int)


# =========================================================
# 3. Swiss Roll Dataset
# =========================================================
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

    rng = np.random.RandomState(random_state)
    extra_dims = rng.normal(
        0, 1,
        size=(n_samples, n_features - 3)
    )

    X = np.concatenate([X_3d, extra_dims], axis=1)

    y = (t > np.median(t)).astype(int)

    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), y.astype(int)


# =========================================================
# 4. Helix / Spiral Dataset
# =========================================================
def generate_helix_dataset(
    n_samples=500,
    n_features=10,
    noise=0.1,
    random_state=42
):
    rng = np.random.RandomState(random_state)

    t = np.linspace(0, 4 * np.pi, n_samples)

    x = np.cos(t)
    y = np.sin(t)
    z = t / (4 * np.pi)

    X_3d = np.vstack([x, y, z]).T

    X_3d += noise * rng.normal(size=X_3d.shape)

    extra_dims = rng.normal(
        0, 1,
        size=(n_samples, n_features - 3)
    )

    X = np.concatenate([X_3d, extra_dims], axis=1)

    labels = (t > 2 * np.pi).astype(int)

    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), labels.astype(int)


# =========================================================
# 5. Torus Dataset
# =========================================================
def generate_torus_dataset(
    n_samples=500,
    n_features=10,
    R=2.0,
    r=0.7,
    random_state=42
):
    rng = np.random.RandomState(random_state)

    theta = rng.uniform(0, 2*np.pi, n_samples)
    phi = rng.uniform(0, 2*np.pi, n_samples)

    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    X_3d = np.vstack([x, y, z]).T

    extra_dims = rng.normal(
        0, 1,
        size=(n_samples, n_features - 3)
    )

    X = np.concatenate([X_3d, extra_dims], axis=1)

    labels = (phi > np.pi).astype(int)

    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), labels.astype(int)


# =========================================================
# 6. Möbius Strip Dataset
# =========================================================
def generate_mobius_dataset(
    n_samples=500,
    n_features=10,
    random_state=42
):
    rng = np.random.RandomState(random_state)

    u = rng.uniform(0, 2*np.pi, n_samples)
    v = rng.uniform(-0.5, 0.5, n_samples)

    x = (1 + v * np.cos(u / 2)) * np.cos(u)
    y = (1 + v * np.cos(u / 2)) * np.sin(u)
    z = v * np.sin(u / 2)

    X_3d = np.vstack([x, y, z]).T

    extra_dims = rng.normal(
        0, 1,
        size=(n_samples, n_features - 3)
    )

    X = np.concatenate([X_3d, extra_dims], axis=1)

    labels = (u > np.pi).astype(int)

    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), labels.astype(int)


# =========================================================
# 7. Lorenz Attractor Dataset
# =========================================================
def generate_lorenz_dataset(
    n_samples=500,
    n_features=10,
    sigma=10.0,
    beta=8/3,
    rho=28.0,
    dt=0.01,
    random_state=42
):
    rng = np.random.RandomState(random_state)

    xs = np.zeros(n_samples)
    ys = np.zeros(n_samples)
    zs = np.zeros(n_samples)

    xs[0], ys[0], zs[0] = (0.1, 0.0, 0.0)

    for i in range(n_samples - 1):
        x, y, z = xs[i], ys[i], zs[i]

        xs[i+1] = x + sigma*(y - x)*dt
        ys[i+1] = y + (x*(rho - z) - y)*dt
        zs[i+1] = z + (x*y - beta*z)*dt

    X_3d = np.vstack([xs, ys, zs]).T

    extra_dims = rng.normal(
        0, 1,
        size=(n_samples, n_features - 3)
    )

    X = np.concatenate([X_3d, extra_dims], axis=1)

    labels = (zs > np.median(zs)).astype(int)

    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), labels.astype(int)


# =========================================================
# 8. Two Moons Dataset
# =========================================================
def generate_two_moons_dataset(
    n_samples=500,
    n_features=10,
    noise=0.1,
    random_state=42
):
    X_2d, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )

    rng = np.random.RandomState(random_state)

    extra_dims = rng.normal(
        0, 1,
        size=(n_samples, n_features - 2)
    )

    X = np.concatenate([X_2d, extra_dims], axis=1)

    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), y.astype(int)


# =========================================================
# 9. Sierpinski Triangle Dataset
# =========================================================
def generate_sierpinski_dataset(
    n_samples=500,
    n_features=10,
    random_state=42
):
    rng = np.random.RandomState(random_state)

    vertices = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ])

    points = np.zeros((n_samples, 2))

    current = np.array([0.0, 0.0])

    for i in range(n_samples):
        vertex = vertices[rng.randint(0, 3)]
        current = (current + vertex) / 2
        points[i] = current

    labels = (points[:, 1] > 0.3).astype(int)

    extra_dims = rng.normal(
        0, 1,
        size=(n_samples, n_features - 2)
    )

    X = np.concatenate([points, extra_dims], axis=1)

    X = StandardScaler().fit_transform(X)

    return X.astype(np.float32), labels.astype(int)


# =========================================================
# Save all datasets
# =========================================================
def save_datasets():

    datasets = {
        "gaussian": generate_gaussian_clusters(n_features=10),
        "noise": generate_random_noise(n_features=20),
        "swiss_roll": generate_swiss_roll_50d(n_features=50),
        "helix": generate_helix_dataset(n_features=20),
        "torus": generate_torus_dataset(n_features=20),
        "mobius": generate_mobius_dataset(n_features=10),
        "lorenz": generate_lorenz_dataset(n_features=30),
        "two_moons": generate_two_moons_dataset(n_features=100),
        "sierpinski": generate_sierpinski_dataset(n_features=50),
    }

    for name, (X, y) in datasets.items():

        dir_path = f"data/{name}/raw"

        os.makedirs(dir_path, exist_ok=True)

        np.save(f"{dir_path}/X.npy", X)
        np.save(f"{dir_path}/y.npy", y)

        print(f"Saved {name}: {X.shape} → {dir_path}")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    save_datasets()
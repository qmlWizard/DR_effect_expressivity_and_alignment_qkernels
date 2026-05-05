# dr_methods.py

import numpy as np

# sklearn
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# UMAP
import umap



# -----------------------------
# 1. PCA
# -----------------------------
def dr_pca(X, d_out):
    return PCA(n_components=d_out).fit_transform(X)


# -----------------------------
# 2. Random Projection
# -----------------------------
def dr_random_projection(X, d_out):
    rp = GaussianRandomProjection(n_components=d_out)
    return rp.fit_transform(X)


# -----------------------------
# 3. UMAP
# -----------------------------
def dr_umap(X, d_out):
    reducer = umap.UMAP(n_components=d_out, random_state=42)
    return reducer.fit_transform(X)


# -----------------------------
# 4. Feature Selection (Mutual Info)
# -----------------------------
def dr_feature_selection(X, y, d_out):
    selector = SelectKBest(mutual_info_classif, k=d_out)
    return selector.fit_transform(X, y)


def dr_autoencoder(X, d_out, epochs=50, lr=1e-3, batch_size=32):
    device = torch.device("cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=X.shape[1], latent_dim=d_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = loss_fn(x_hat, batch)
            loss.backward()
            optimizer.step()

    # Extract latent representation
    model.eval()
    with torch.no_grad():
        _, Z = model(X_tensor)

    return Z.cpu().numpy()


# -----------------------------
# 6. Normalization for Quantum
# -----------------------------
def normalize_to_quantum_range(X):
    scaler = MinMaxScaler(feature_range=(-np.pi/2, np.pi/2))
    return scaler.fit_transform(X)


# -----------------------------
# 7. Unified Interface
# -----------------------------
def apply_dr(X, y, method, d_out):
    if method == "pca":
        X_red = dr_pca(X, d_out)

    elif method == "rp":
        X_red = dr_random_projection(X, d_out)

    elif method == "umap":
        X_red = dr_umap(X, d_out)

    elif method == "fs":
        X_red = dr_feature_selection(X, y, d_out)

    elif method == "ae":
        X_red = dr_autoencoder(X, d_out)

    else:
        raise ValueError(f"Unknown DR method: {method}")

    # Final normalization step for quantum encoding
    X_red = normalize_to_quantum_range(X_red)

    return X_red
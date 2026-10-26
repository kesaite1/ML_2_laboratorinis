
import numpy as np
import pandas as pd
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances


# Duomenų paruošimas
df = pd.read_csv("A24_cleaned.csv")
X = df.drop(columns=["class"]).select_dtypes(include="number")
X_scaled = StandardScaler().fit_transform(X)


def continuity(X_original, X_embedded, k=15):
    n = len(X_original)

    D_orig = pairwise_distances(X_original)
    D_emb = pairwise_distances(X_embedded)

    np.fill_diagonal(D_orig, np.inf)
    np.fill_diagonal(D_emb, np.inf)

    order_orig = np.argsort(D_orig, axis=1)
    order_emb = np.argsort(D_emb, axis=1)

    ranks_emb = np.empty_like(order_emb)
    ranks_emb[np.arange(n)[:, None], order_emb] = np.arange(n)

    neighbors_orig = order_orig[:, :k]

    mask_emb = np.zeros((n, n), dtype=bool)
    mask_emb[np.arange(n)[:, None], order_emb[:, :k]] = True

    missing = ~mask_emb[np.arange(n)[:, None], neighbors_orig]
    penalties = np.maximum(ranks_emb - k, 0)

    total = penalties[
        np.arange(n)[:, None], neighbors_orig
    ][missing].sum()

    denominator = n * k * (2 * n - 3 * k - 1)
    return 1 - (2 * total / denominator)



results = []

for min_dist in [0.0, 0.1, 0.5]:
    model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=min_dist,
        random_state=42,
        n_jobs=1
    )

    embedding = model.fit_transform(X_scaled)

    T = trustworthiness(X_scaled, embedding, n_neighbors=15)
    C = continuity(X_scaled, embedding, k=15)

    results.append({
        "min_dist": min_dist,
        "Trustworthiness": T,
        "Continuity": C,
        "|T-C|": abs(T - C)
    })

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))
results_df.to_csv("umap_min_dist_results.csv", index=False)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

# 1. Load the cleaned dataset
df = pd.read_csv("A24_cleaned.csv")

# 2. Separate features from the class label
y = df["class"]
X = df.drop(columns=["class", "Outside_Global_Index"])

# 3. Keep only numeric features and check the data
X = X.select_dtypes(include=np.number)
X = X.replace([np.inf, -np.inf], np.nan)

valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate original-space pairwise distances once
# Each pair of observations is included only once
original_distances = pdist(X_scaled, metric="euclidean")

# Denominator for normalized Stress, according to the course formula
stress_denominator = np.sum(original_distances ** 2)



def continuity(X, embedding, n_neighbors=15):
    """
    Calculate Continuity (C) without storing an n x n matrix.
    """
    n_samples = X.shape[0]
    k = n_neighbors

    # Find k nearest neighbors in the original space
    original_nn = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(X)

    original_indices = original_nn.kneighbors(
        X, return_distance=False
    )[:, 1:]

    # Find k nearest neighbors in the embedded space
    embedded_nn = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(embedding)

    embedded_indices = embedded_nn.kneighbors(
        embedding, return_distance=False
    )[:, 1:]

    penalty = 0

    for i in range(n_samples):
        embedded_set = set(embedded_indices[i])

        # Original neighbors missing from the embedding's k-neighborhood
        missing_neighbors = [
            j for j in original_indices[i]
            if j not in embedded_set
        ]

        if not missing_neighbors:
            continue

        # Rank these points by distance in the embedded space
        distances = np.linalg.norm(
            embedding - embedding[i], axis=1
        )

        order = np.argsort(distances)
        ranks = np.empty(n_samples, dtype=np.int32)
        ranks[order] = np.arange(n_samples)

        # Rank 0 is the point itself; convert other ranks to 1-based
        ranks[order[order != i]] -= 1
        ranks[i] = 0

        for j in missing_neighbors:
            penalty += max(0, ranks[j] - k)

    denominator = n_samples * k * (
        2 * n_samples - 3 * k - 1
    )

    return 1 - (2 * penalty / denominator)




# 5. Compare UMAP parameter combinations
results = []

neighbor_values = [5, 15, 50]
min_dist_values = [0.0, 0.1, 0.5]

fig, axes = plt.subplots(3, 3, figsize=(15, 13))

for i, n_neighbors in enumerate(neighbor_values):
    for j, min_dist in enumerate(min_dist_values):

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )

        embedding = reducer.fit_transform(X_scaled)

        # 6. Evaluate local neighborhood preservation
        trust_scores = {}

        for k in [15]:
            score = trustworthiness(
                X_scaled,
                embedding,
                n_neighbors=k
            )
            trust_scores[k] = score

            continuity_score = continuity(
                X_scaled,
                embedding,
                n_neighbors=15
            )

        # 7. Calculate global distance-preservation metrics
        embedded_distances = pdist(
            embedding,
            metric="euclidean"
        )

        # Spearman rank correlation coefficient (rho)
        rho = spearmanr(
            original_distances,
            embedded_distances
        ).statistic

        # Normalized Stress according to the course formula:
        # sum((d_original - d_embedded)^2) / sum(d_original^2)
        stress_1 = (
            np.sum((original_distances - embedded_distances) ** 2)
            / stress_denominator
        )

        results.append({
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            ##"Trustworthiness k=5": trust_scores[5],
            "Trustworthiness k=15": trust_scores[15],
            ##"Trustworthiness k=50": trust_scores[50],
            "Continuity k=15": continuity_score,
            "Spearman rho": rho,
            "Stress-1": stress_1
        })

        # 8. Plot each configuration
        ax = axes[i, j]

        for label in sorted(y.unique()):
            mask = (y == label).to_numpy()

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=8,
                alpha=0.65,
                label=str(label)
            )

        ax.set_title(
            f"n_neighbors={n_neighbors}, min_dist={min_dist}"
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

plt.tight_layout()

# 9. Display and save the comparison table
results_df = pd.DataFrame(results)

print("\nUMAP parameter comparison:")
print(results_df.round(4).to_string(index=False))

# Save CSV BEFORE handling plots
results_df.to_csv("umap_parameter_results.csv", index=False)
print("\nSaved results to umap_parameter_results.csv")

# Save the plot as an image instead of waiting for a window
plt.tight_layout()
plt.savefig(
    "umap_parameter_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Saved plots to umap_parameter_comparison.png")
print("\nAnalysis completed successfully.")

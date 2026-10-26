
import numpy as np
import pandas as pd
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances


# 1. Duomenų paruošimas
df = pd.read_csv("A24_cleaned.csv")

# Klasių žymos nėra naudojamos UMAP mokymui
X = df.drop(columns=["class"]).select_dtypes(include="number")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Objektų skaičius: {X_scaled.shape[0]}")
print(f"Požymių skaičius: {X_scaled.shape[1]}")


# 2. Continuity skaičiavimo funkcija
def calculate_continuity(X_original, X_embedded, k=50):
    n = X_original.shape[0]

    # Atstumai pradinėje ir sumažintoje erdvėje
    dist_original = pairwise_distances(X_original)
    dist_embedded = pairwise_distances(X_embedded)

    # Paties objekto nelaikome savo kaimynu
    np.fill_diagonal(dist_original, np.inf)
    np.fill_diagonal(dist_embedded, np.inf)

    # Kiekvieno objekto kaimynų rangai
    order_original = np.argsort(dist_original, axis=1)
    order_embedded = np.argsort(dist_embedded, axis=1)

    ranks_embedded = np.empty_like(order_embedded)
    ranks_embedded[
        np.arange(n)[:, None], order_embedded
    ] = np.arange(n)[None, :]

    # Pradinių k kaimynų aibės
    original_neighbors = order_original[:, :k]

    # Sumažintoje erdvėje esančių k kaimynų žymėjimas
    embedded_neighbor_mask = np.zeros((n, n), dtype=bool)
    embedded_neighbor_mask[
        np.arange(n)[:, None], order_embedded[:, :k]
    ] = True

    # Tikri kaimynai, kurių nebeliko tarp k artimiausių
    missing_neighbors = ~embedded_neighbor_mask[
        np.arange(n)[:, None], original_neighbors
    ]

    penalties = np.maximum(ranks_embedded - k, 0)
    total_penalty = penalties[
        np.arange(n)[:, None], original_neighbors
    ][missing_neighbors].sum()

    denominator = n * k * (2 * n - 3 * k - 1)

    return 1 - (2 * total_penalty / denominator)


# 3. Palyginame tikslinių dimensijų skaičių
results = []

for d in [2]:
    reducer = umap.UMAP(
        n_components=d,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=1
    )

    embedding = reducer.fit_transform(X_scaled)

    T = trustworthiness(
        X_scaled, embedding, n_neighbors=15
    )

    C = calculate_continuity(
        X_scaled, embedding, k=50
    )

    results.append({
        "Dimensijos (d)": d,
        "Trustworthiness (T)": T,
        "Continuity (C)": C,
        "|T-C|": abs(T - C)
    })

    print(f"\nDimensijos: {d}")
    print(f"Trustworthiness (k=15): {T:.4f}")
    print(f"Continuity (k=15):      {C:.4f}")
    print(f"|T-C|:                  {abs(T-C):.4f}")


# 4. Rezultatų lentelė
results_df = pd.DataFrame(results)
print("\nGALUTINĖ REZULTATŲ LENTELĖ")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "umap_dimensions_results.csv",
    index=False
)


from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


# --------------------------------------------------
# 1. Nustatymai
# --------------------------------------------------

FILES = {
    "A": "A24_cleaned.csv",
    "B": "A24_no_redundant.csv",
    "C": "A24_no_outliers.csv",
}

OUTPUT_DIR = Path("umap_results_ABC")
OUTPUT_DIR.mkdir(exist_ok=True)

N_COMPONENTS = 2
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42

K_VALUES = [5, 15, 50]


# --------------------------------------------------
# 2. KNN kaimynų išsaugojimo rodiklis
# --------------------------------------------------

def knn_preservation(X, embedding, k):
    original_nn = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(X)

    embedded_nn = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(embedding)

    original_indices = original_nn.kneighbors(
        return_distance=False
    )[:, 1:]

    embedded_indices = embedded_nn.kneighbors(
        return_distance=False
    )[:, 1:]

    scores = []

    for original, embedded in zip(
        original_indices, embedded_indices
    ):
        scores.append(
            len(set(original) & set(embedded)) / k
        )

    return float(np.mean(scores))


# --------------------------------------------------
# 3. Tęstinumo (Continuity) metrika
# --------------------------------------------------

def continuity(X, embedding, k):
    """
    Apskaičiuoja tęstinumą (Continuity).

    Vertina, kiek pradinių k artimiausių kaimynų
    išsaugoma projekcijoje, bausdama už prarastus
    pradinius kaimynus.

    Reikšmė artėja prie 1, kai kaimynystė
    pradinėje erdvėje gerai išsaugoma projekcijoje.
    """

    n = X.shape[0]

    if k >= n / 2:
        raise ValueError(
            "k turi būti mažesnis už objektų skaičiaus pusę."
        )

    # K artimiausi kaimynai pradinėje erdvėje.
    original_nn = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(X)

    original_indices = original_nn.kneighbors(
        return_distance=False
    )[:, 1:]

    # Visų objektų atstumai projekcijos erdvėje.
    embedded_distances = pairwise_distances(
        embedding, metric="euclidean"
    )

    # Rangas: kiek objektų yra arčiau už kiekvieną objektą.
    # Pašaliname savęs atstumą, kuris lygus nuliui.
    embedded_ranks = np.argsort(
        np.argsort(embedded_distances, axis=1),
        axis=1
    )

    penalty_sum = 0

    for i in range(n):
        original_set = set(original_indices[i])

        # Projekcijos k artimiausi kaimynai.
        embedded_set = set(
            np.argsort(embedded_distances[i])[1:k + 1]
        )

        # Kaimynai, kurie buvo arti pradinėje erdvėje,
        # bet nepateko į projekcijos k artimiausiųjų sąrašą.
        missing_neighbors = original_set - embedded_set

        for j in missing_neighbors:
            penalty_sum += embedded_ranks[i, j] - k

    denominator = n * k * (2 * n - 3 * k - 1)

    return float(
        1 - (2 * penalty_sum / denominator)
    )


# --------------------------------------------------
# 4. UMAP A, B ir C rinkiniams
# --------------------------------------------------

all_metrics = []

for dataset_name, file_path in FILES.items():

    print(f"\n{'=' * 45}")
    print(f"Rinkinys {dataset_name}: {file_path}")

    df = pd.read_csv(file_path)

    if "class" not in df.columns:
        raise ValueError(
            f"{file_path} nerastas 'class' stulpelis."
        )

    # Klasė naudojama tik grafikų spalvoms.
    y = df["class"].astype(str)

    # Klasės etiketė neįtraukiama į UMAP.
    X_df = (
        df.drop(columns=["class"])
        .select_dtypes(include="number")
    )

    if X_df.isna().any().any():
        raise ValueError(
            f"{dataset_name} rinkinyje yra trūkstamų reikšmių."
        )

    X = StandardScaler().fit_transform(X_df)

    print(f"Objektų: {X.shape[0]}")
    print(f"Požymių: {X.shape[1]}")
    print("Vykdomas UMAP...")

    reducer = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    embedding = reducer.fit_transform(X)

    # ----------------------------------------------
    # 5. Grafikas
    # ----------------------------------------------

    plt.figure(figsize=(10, 7))

    for class_name in sorted(y.unique()):
        mask = y.to_numpy() == class_name

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=0.7,
            label=class_name,
        )

    plt.title(
        f"UMAP — rinkinys {dataset_name}\n"
        f"n_neighbors={N_NEIGHBORS}, "
        f"min_dist={MIN_DIST}"
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(
        title="Klasė",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"umap_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grafikas išsaugotas: {plot_path}")

    # ----------------------------------------------
    # 6. Kokybės metrikos
    # ----------------------------------------------

    original_distances = pdist(X, metric="euclidean")

    # Atstumai tarp tų pačių objektų porų UMAP projekcijoje
    embedded_distances = pdist(embedding, metric="euclidean")

    # Spearmano koreliacijos koeficientas (rho)
    rho = spearmanr(
        original_distances,
        embedded_distances
    ).statistic

    # Stress-1: atstumų iškraipymo rodiklis
    stress_1 = (
        np.sum((original_distances - embedded_distances) ** 2)
        / np.sum(original_distances ** 2)
    )

    print(
        f"Globalios struktūros rodikliai: "
        f"rho={rho:.4f}, Stress-1={stress_1:.4f}"
    )

    # Lokalių kaimynysčių išsaugojimo rodikliai
    for k in K_VALUES:
        tw = trustworthiness(X, embedding, n_neighbors=k)
        cont = continuity(X, embedding, k)
        knn = knn_preservation(X, embedding, k)

        all_metrics.append({
            "Dataset": dataset_name,
            "Objects": len(df),
            "Features": X.shape[1],
            "k": k,
            "Trustworthiness": tw,
            "Continuity": cont,
            "KNN_preservation": knn,
            "Spearman_rho": rho,
            "Stress-1": stress_1,
        })

        print(
            f"k={k}: "
            f"T={tw:.4f}, "
            f"C={cont:.4f}, "
            f"KNN={knn:.4f}, "
            f"rho={rho:.4f}, "
            f"Stress-1={stress_1:.4f}"
        )
# --------------------------------------------------
# 7. Išsaugome bendrą rezultatų lentelę
# --------------------------------------------------

results = pd.DataFrame(all_metrics)

results_path = OUTPUT_DIR / "metrics_ABC.csv"
results.to_csv(results_path, index=False)

print("\n--- BENDRI REZULTATAI ---")
print(results.round(4).to_string(index=False))
print(f"\nMetrikos išsaugotos: {results_path}")


# --------------------------------------------------
# 8. A ir C palyginimas su bendrais objektais
# --------------------------------------------------

print("\n--- A IR C PALYGINIMAS SU BENDRAIS OBJEKTAIS ---")

df_A = pd.read_csv(FILES["A"]).reset_index(drop=True)
df_C = pd.read_csv(FILES["C"]).reset_index(drop=True)

# Sulyginame požymių stulpelius. Klasės etiketė nenaudojama
# objektams sutapatinti, nes ji nėra UMAP požymis.
X_A_df = (
    df_A.drop(columns=["class"])
    .select_dtypes(include="number")
)
X_C_df = (
    df_C.drop(columns=["class"])
    .select_dtypes(include="number")
)

if list(X_A_df.columns) != list(X_C_df.columns):
    raise ValueError(
        "A ir C aibių požymių stulpeliai arba jų tvarka skiriasi. "
        "Prieš lyginant reikia sulyginti požymius."
    )

if X_A_df.isna().any().any() or X_C_df.isna().any().any():
    raise ValueError("A arba C aibėje yra trūkstamų reikšmių.")

# Sutapatiname C eilutes su A eilutėmis pagal visų skaitinių
# požymių reikšmes. Tai veikia, jei C gauta iš A pašalinant eilutes.
# Naudojame mažą toleranciją dėl galimų CSV apvalinimo skirtumų.
A_values = X_A_df.to_numpy(dtype=float)
C_values = X_C_df.to_numpy(dtype=float)

indices_A = []
used_A_indices = set()
unmatched_C_rows = []
ambiguous_C_rows = []
TOLERANCE = 1e-8

for c_idx, c_row in enumerate(C_values):
    # Ieškome A eilučių, kurių visi požymiai sutampa su C eilute.
    matches = np.flatnonzero(
        np.all(np.isclose(A_values, c_row, rtol=0.0, atol=TOLERANCE), axis=1)
    )

    # Jei yra keli sutapimai, pirmenybę teikiame dar nepanaudotam indeksui.
    available_matches = [int(i) for i in matches if int(i) not in used_A_indices]

    if len(available_matches) == 0:
        unmatched_C_rows.append(c_idx)
        indices_A.append(None)
    else:
        if len(available_matches) > 1:
            ambiguous_C_rows.append(c_idx)
        chosen = available_matches[0]
        indices_A.append(chosen)
        used_A_indices.add(chosen)

if unmatched_C_rows:
    print(
        f"Nepavyko sutapatinti {len(unmatched_C_rows)} C eilučių pagal "
        "skaitinius požymius. Patikrink, ar C tikrai sukurta iš A, "
        "nekeičiant požymių reikšmių."
    )
    print("Pirmos nesutapatintos C eilučių pozicijos:", unmatched_C_rows[:10])
    raise ValueError(
        "A/C bendrų objektų palyginimas nutrauktas, kad nebūtų "
        "pateikti neteisingi rezultatai. Patikimiausias sprendimas – "
        "išskirčių šalinimo metu išsaugoti originalius A eilučių indeksus."
    )

indices_A = np.asarray(indices_A, dtype=int)

if len(indices_A) != len(df_C) or len(np.unique(indices_A)) != len(indices_A):
    raise ValueError(
        "Objektų sutapatinimas nėra vienareikšmis. Reikia išskirčių "
        "šalinimo metu išsaugoti originalius eilučių indeksus."
    )

if ambiguous_C_rows:
    print(
        f"Pastaba: {len(ambiguous_C_rows)} C eilučių turėjo kelis vienodus "
        "požymių atitikmenis A aibėje. Patikrink dublikatus, jei tokių yra."
    )

print(f"Objektų A aibėje: {len(df_A)}")
print(f"Objektų C aibėje: {len(df_C)}")
print(f"Sutapatintų bendrų objektų: {len(indices_A)}")

# SVARBU: standartizatorių pritaikome A aibei, o tą patį transformavimą
# taikome C aibei. Taip atskiriame eilučių pašalinimo poveikį nuo
# papildomo standartizavimo parametrų pasikeitimo.
scaler_common = StandardScaler()
X_A = scaler_common.fit_transform(X_A_df)
X_C = scaler_common.transform(X_C_df)

# A projekcija skaičiuojama su visais A objektais.
embedding_A = umap.UMAP(
    n_components=N_COMPONENTS,
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    random_state=RANDOM_STATE,
    n_jobs=1,
).fit_transform(X_A)

# C projekcija skaičiuojama tik su išlikusiais objektais.
embedding_C = umap.UMAP(
    n_components=N_COMPONENTS,
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    random_state=RANDOM_STATE,
    n_jobs=1,
).fit_transform(X_C)

# Iš A projekcijos atrenkame tuos pačius objektus, kurie išliko C aibėje.
embedding_A_common = embedding_A[indices_A]
embedding_C_common = embedding_C

# Palyginami atstumai tarp tų pačių objektų porų abiejose projekcijose.
distances_A = pdist(embedding_A_common, metric="euclidean")
distances_C = pdist(embedding_C_common, metric="euclidean")

rho_common = spearmanr(distances_A, distances_C).statistic
stress_denominator = np.sum(distances_A ** 2)
if stress_denominator == 0:
    raise ValueError("Negalima apskaičiuoti Stress-1: A projekcijos atstumų suma lygi nuliui.")

stress_common = (
    np.sum((distances_A - distances_C) ** 2) / stress_denominator
)

print(f"Spearman rho tarp A ir C projekcijų: {rho_common:.4f}")
print(f"Stress-1 tarp A ir C projekcijų: {stress_common:.4f}")

# Kaimynysčių sutapimas: kiekvienam objektui lyginame kaimynus
# tarp tų pačių bendrų objektų abiejose projekcijose.
knn_common_results = []

for k in K_VALUES:
    nn_A = NearestNeighbors(n_neighbors=k + 1).fit(embedding_A_common)
    nn_C = NearestNeighbors(n_neighbors=k + 1).fit(embedding_C_common)

    neighbors_A = nn_A.kneighbors(return_distance=False)[:, 1:]
    neighbors_C = nn_C.kneighbors(return_distance=False)[:, 1:]

    overlaps = [
        len(set(a) & set(c)) / k
        for a, c in zip(neighbors_A, neighbors_C)
    ]

    overlap_mean = float(np.mean(overlaps))
    knn_common_results.append({
        "k": k,
        "Neighborhood_overlap": overlap_mean,
    })
    print(f"k={k}: bendrų kaimynų sutapimas={overlap_mean:.4f}")

common_results = pd.DataFrame([{
    "Objects_A": len(df_A),
    "Objects_C": len(df_C),
    "Common_objects": len(indices_A),
    "Spearman_rho_between_embeddings": rho_common,
    "Stress-1_between_embeddings": stress_common,
}])

common_results.to_csv(OUTPUT_DIR / "common_objects_A_C.csv", index=False)
pd.DataFrame(knn_common_results).to_csv(
    OUTPUT_DIR / "common_neighbors_A_C.csv", index=False
)

print("Bendrų objektų rezultatai išsaugoti:")
print(OUTPUT_DIR / "common_objects_A_C.csv")
print(OUTPUT_DIR / "common_neighbors_A_C.csv")

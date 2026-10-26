import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_PATH = "A24_cleaned.csv"
OUTPUT_PATH = "A24_no_redundant.csv"
TARGET_COLUMN = "class"

CORRELATION_METHOD = "spearman"
REDUNDANCY_THRESHOLD = 0.9
PREFERRED_PREFIXES = ("Log",)
TOP_N = 20

HEATMAP_SIZE = (14, 10)
HEATMAP_CMAP = "coolwarm"

"""
Kodas palieka log-transformuotas versija, nes šie požymiai yra geriau pasiskirstę, tai jiems paliekam pirmenybę.
Pašalinami tik pagrįstai pertekliniais laikomi stipriai koreliuoti požymiai.
"""

def numeric_features(df, target_column=TARGET_COLUMN):
    return df.drop(columns=target_column).select_dtypes(include=[np.number])


def correlation_matrix(features, method=CORRELATION_METHOD):
    return features.corr(method=method)


def unique_pairs(corr):
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper_triangle.stack().reset_index()
    pairs.columns = ["Column_1", "Column_2", "Correlation"]
    return pairs


def most_correlated_pairs(corr, top_n=TOP_N):
    pairs = unique_pairs(corr)
    pairs["Abs_Correlation"] = pairs["Correlation"].abs()
    ranked = pairs.sort_values("Abs_Correlation", ascending=False, ignore_index=True)
    return ranked.head(top_n)


def preferred_features(columns, prefixes=PREFERRED_PREFIXES):
    return {column for column in columns if column.startswith(prefixes)}


def greedy_redundancy_filter(corr, threshold=REDUNDANCY_THRESHOLD, preferred=None):
    
    if preferred is None:
        preferred = preferred_features(corr.columns)

    abs_corr = corr.abs()

    remaining = list(corr.columns)
    removed = []

    while len(remaining) > 1:
        block = abs_corr.loc[remaining, remaining]
        upper = block.where(np.triu(np.ones(block.shape), k=1).astype(bool)).stack()
        if upper.empty or upper.max() < threshold:
            break

        first, second = upper.idxmax()
        if (first in preferred) != (second in preferred):
            redundant = second if first in preferred else first
        else:
            off_diagonal = block.where(
                ~np.eye(len(remaining), dtype=bool)
            )
            mean_to_others = off_diagonal.sum() / (len(remaining) - 1)
            redundant = first if mean_to_others[first] >= mean_to_others[second] else second

        partner = first if redundant == second else second
        removed.append((redundant, partner, corr.at[redundant, partner]))
        remaining.remove(redundant)

    return remaining, removed


def plot_correlation_heatmap(corr, title="Numeric feature correlation matrix"):
    plt.figure(figsize=HEATMAP_SIZE)
    sns.heatmap(corr, cmap=HEATMAP_CMAP, center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


df = pd.read_csv(INPUT_PATH)
features = numeric_features(df)

corr = correlation_matrix(features)
pairs = unique_pairs(corr)

redundant_pairs = pairs[pairs["Correlation"].abs() >= REDUNDANCY_THRESHOLD]
print(f"Feature pairs with |r| >= {REDUNDANCY_THRESHOLD}: {len(redundant_pairs)}")

kept_features, removed_records = greedy_redundancy_filter(corr)
print(f"Features removed by greedy rule: {len(removed_records)}")
removed_table = pd.DataFrame(removed_records, columns=["Removed", "Correlated with", "r"])
print(removed_table.to_string(index=False))
print(f"Remaining features: {len(kept_features)}")

reduced = df[kept_features + [TARGET_COLUMN]]
reduced.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH} {reduced.shape}")

after_removal = corr.loc[kept_features, kept_features]

print(f"\nTop {TOP_N} most correlated feature pairs after removal ({CORRELATION_METHOD}):")
print(most_correlated_pairs(after_removal).to_string(index=False))

plot_correlation_heatmap(corr, title=f"Feature correlation matrix before removal ({CORRELATION_METHOD})")
plot_correlation_heatmap(after_removal, title=f"Feature correlation matrix after removal ({CORRELATION_METHOD})")
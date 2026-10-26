import numpy as np
import pandas as pd


INPUT_PATH = "A24_cleaned.csv"
OUTPUT_PATH = "A24_no_outliers.csv"

IQR_MULTIPLIER = 1.5
MIN_FLAGGED_FEATURES = 1


def detection_features(df):
    features = df.drop(columns="class").copy()
    return features


def outlier_flags(df, k=IQR_MULTIPLIER):
    features = detection_features(df)
    q1 = features.quantile(0.25)
    q3 = features.quantile(0.75)
    iqr = q3 - q1
    below_range = features.lt(q1 - k * iqr)
    above_range = features.gt(q3 + k * iqr)
    return below_range | above_range


def rows_with_outliers(flags, min_feats=MIN_FLAGGED_FEATURES):
    return flags.sum(axis=1) >= min_feats


def drop_summary(flags, rows_to_drop):
    counts = flags[rows_to_drop].sum().astype(int)
    share = (counts / rows_to_drop.sum() * 100).round(1)
    summary = pd.DataFrame({
        "Požymis": counts.index,
        "Pašalinta eilučių": counts.to_numpy(),
        "Dalis nuo pašalintų, %": share.to_numpy(),
    })
    return summary.sort_values("Pašalinta eilučių", ascending=False, ignore_index=True)


df = pd.read_csv(INPUT_PATH)
flags = outlier_flags(df)
rows_to_drop = rows_with_outliers(flags)

kept = df[~rows_to_drop]
kept.to_csv(OUTPUT_PATH, index=True, index_label = "index_new")

print(f"{len(df)} -> {len(kept)} (pašalinta {rows_to_drop.sum()}, {rows_to_drop.mean() * 100:.1f} %)")
print(f"\nPašalintos eilutės pagal požymį (k={IQR_MULTIPLIER}, min. požymių eilutėje={MIN_FLAGGED_FEATURES}):")
print(drop_summary(flags, rows_to_drop).to_string(index=False))
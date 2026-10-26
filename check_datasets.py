
import pandas as pd

files = {
    "A — pradinis": "A24_cleaned.csv",
    "B — be perteklinių požymių": "A24_no_redundant.csv",
    "C — be išskirčių": "A24_no_outliers.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)

    features = df.drop(
        columns=["class"],
        errors="ignore"
    ).select_dtypes(include="number")

    print(f"\n--- {name} ---")
    print(f"Eilučių skaičius: {len(df)}")
    print(f"Požymių skaičius: {features.shape[1]}")
    print(f"Trūkstamų reikšmių: {df.isna().sum().sum()}")
    print(f"Pasikartojančių eilučių: {df.duplicated().sum()}")

    if "class" in df.columns:
        print("Klasių pasiskirstymas:")
        print(df["class"].value_counts().sort_index().to_dict())
    else:
        print("DĖMESIO: nėra 'class' stulpelio!")

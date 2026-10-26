import pandas as pd
import numpy as np
import matplotlib.pylab as plt

MY_COLS_EM = ["Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer", "Steel_Plate_Thickness",
              "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index"]

NUMERIC_COLS = [
    "X_Minimum",
    "X_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
]

MY_COLS = [
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """1) Pašalinami dublikatai."""
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df)
    # print(f"1) Pašalinta dublikatų: {duplicates_removed}")
    return df


def swap_x_min_max(df: pd.DataFrame) -> pd.DataFrame:
    """2) Sukeičiamos blogos X_Minimum ir X_Maximum."""
    mask_x = df["X_Minimum"] > df["X_Maximum"]
    x_swapped = mask_x.sum()
    df.loc[mask_x, ["X_Minimum", "X_Maximum"]] = (
        df.loc[mask_x, ["X_Maximum", "X_Minimum"]].to_numpy()
    )
    # print(f"2) Sukeista X_Minimum > X_Maximum: {x_swapped}")
    return df


def abs_pixels_areas(df: pd.DataFrame) -> pd.DataFrame:
    """3) abs(Pixels_Areas), jei < 0."""
    mask_neg = df["Pixels_Areas"] < 0
    negative_areas = mask_neg.sum()
    df["Pixels_Areas"] = df["Pixels_Areas"].abs()
    # print(f"3) Pakeista neigiamų Pixels_Areas: {negative_areas}")
    return df


def fix_perimeter_area(df: pd.DataFrame) -> pd.DataFrame:
    """4) X_Perimeter * Y_Perimeter < Pixels_Areas keičiama į klasės medianą."""
    mask_perim = (df["X_Perimeter"] * df["Y_Perimeter"]) < df["Pixels_Areas"]
    perimeter_fixed = mask_perim.sum()

    median_by_class = (
        df.loc[~mask_perim]
        .groupby("class")["Pixels_Areas"]
        .median()
    )

    df.loc[mask_perim, "Pixels_Areas"] = (
        df.loc[mask_perim, "class"].map(median_by_class)
    )
    # print(f"4) Pakeista X_Perimeter * Y_Perimeter < Pixels_Areas: {perimeter_fixed}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Vykdo visas valymo pakopas nuosekliai."""
    steps = [
        remove_duplicates,
        swap_x_min_max,
        abs_pixels_areas,
        fix_perimeter_area,
    ]

    for step in steps:
        df = step(df)

    return df

raw = pd.read_csv("A24/A24.csv")

df = raw.copy()

df["Steel_Plate_Thickness"] = df["Steel_Plate_Thickness"].str.replace(" mm", "", regex=False).astype(int)
as_number = pd.to_numeric(raw["Empty_Index"], errors="coerce")
df["Empty_Index"] = as_number

rules = {
    "Šviesumas ne intervale [0; 255]":
        (df.Minimum_of_Luminosity < 0) | (df.Minimum_of_Luminosity > 255) |
        (df.Maximum_of_Luminosity < 0) | (df.Maximum_of_Luminosity > 255),
    "Minimum_of_Luminosity > Maximum_of_Luminosity": df.Minimum_of_Luminosity > df.Maximum_of_Luminosity,
    "Length_of_Conveyer <= 0 arba Steel_Plate_Thickness <= 0":
        (df.Length_of_Conveyer <= 0) | (df.Steel_Plate_Thickness <= 0),
    "Edges_Index < 0 arba > 1": (df.Edges_Index < 0) | (df.Edges_Index > 1),
    "Empty_Index < 0 arba > 1": (df.Empty_Index < 0) | (df.Empty_Index > 1),
    "Square_Index <= 0 arba > 1": (df.Square_Index <= 0) | (df.Square_Index > 1),
    "Outside_X_Index <= 0 arba > 1": (df.Outside_X_Index <= 0) | (df.Outside_X_Index > 1),
}
bounds = pd.Series({name: int(mask.sum()) for name, mask in rules.items()}, name="Pažeidimų skaičius")

dx = df.X_Maximum - df.X_Minimum  # defekto plotis X kryptimi
dy = df.Y_Maximum - df.Y_Minimum  # defekto aukštis Y kryptimi
L = df.Length_of_Conveyer
box = dx * dy  # apgaubiančio stačiakampio plotas
edges_calc = np.minimum(df.X_Minimum, L - df.X_Maximum) / (L / 2)
empty_calc = 1 - df.Pixels_Areas / box
square_calc = np.minimum(dx, dy) / np.maximum(dx, dy)
outx_calc = dx / L
valid_x = dx > 0  # X koordinatės tvarkingos
valid_box = valid_x & (dy > 0) & (df.Pixels_Areas > 0) & (df.Pixels_Areas <= box)

def compare(name, given, calc, valid):
    ok = valid & given.notna()
    diff = (given - calc)[ok].abs()
    return {"Požymis": name, "Patikrinta eilučių": int(ok.sum()),
            "Nesutampa (>0.001)": int((diff > 0.001).sum()), "Didžiausias skirtumas": round(float(diff.max()), 5)}


cross = pd.DataFrame([
    compare("Edges_Index", df.Edges_Index.where(df.Edges_Index <= 1), edges_calc, valid_x),
    compare("Empty_Index", df.Empty_Index, empty_calc, valid_box),
    compare("Square_Index", df.Square_Index, square_calc, valid_x),
    compare("Outside_X_Index", df.Outside_X_Index, outx_calc, valid_x),
])


clean = df.copy()


too_big = clean.Edges_Index > 1
clean.loc[too_big, "Edges_Index"] = np.nan

miss = clean.Edges_Index.isna() & valid_x
clean.loc[miss, "Edges_Index"] = edges_calc[miss].round(4)

miss = clean.Empty_Index.isna() & valid_box
clean.loc[miss, "Empty_Index"] = empty_calc[miss].round(4)

left = clean.Empty_Index.isna()
clean.loc[left, "Empty_Index"] = clean.Empty_Index.median()
# print("Likę trūkstamų reikšmių:", int(clean[MY_COLS_EM].isna().sum().sum()))

df = clean_data(clean)

columns = list(df.columns.values)

columns_lum = columns[:-1]
columns_lum.remove("Luminosity_Index")
df_lum = df[columns_lum].dropna()

df_lum_nan = df[df["Luminosity_Index"].isnull()]
df_lum_nan = df_lum_nan.drop("Luminosity_Index", axis=1)
df_lum_nan = df_lum_nan.drop("class", axis=1)
df_sum = df_lum.apply(np.sum, axis=1, result_type="reduce")


def artimiausia_lum(val, i):
    new_df = df_sum.apply(lambda x: abs(x - val))
    sorted = new_df.sort_values(ascending=True)
    df.at[i, "Luminosity_Index"] = df.at[sorted.index[1], "Luminosity_Index"]


for index, row in df_lum_nan.iterrows():
    artimiausia_lum(row.sum(), index)


columns_sig = columns[:-1]
columns_sig.remove("SigmoidOfAreas")
df_sig = df[columns_sig].dropna()

df_sig_nan = df[df["SigmoidOfAreas"].isnull()]
df_sig_nan = df_sig_nan.drop("SigmoidOfAreas", axis=1)
df_sig_nan = df_sig_nan.drop("class", axis=1)
df_sum = df_sig.apply(np.sum, axis=1, result_type="reduce")


def artimiausia_sig(val, i):
    new_df = df_sum.apply(lambda x: abs(x - val))
    sorted = new_df.sort_values(ascending=True)
    df.at[i, "SigmoidOfAreas"] = df.at[sorted.index[1], "SigmoidOfAreas"]


for index, row in df_sig_nan.iterrows():
    artimiausia_sig(row.sum(), index)


def robust_scaling(column):
    mediana = df[column].median()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df[column] = df[column].apply(lambda x: (x - mediana) / IQR)

columns_new = list(df.columns.values)
columns_new.remove("Outside_Global_Index")
columns_new.remove("class")

# df.boxplot()
# plt.show()

for column in columns_new:
    robust_scaling(column)

df.boxplot()
plt.show()

df.to_csv("A24_cleaned.csv", index=False)

## print(df.info())
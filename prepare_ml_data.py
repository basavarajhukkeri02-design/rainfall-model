
import pandas as pd
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT = BASE_DIR / "karnataka_rainfall_data_2023_2025.csv"
OUTPUT = BASE_DIR / "karnataka_ml_features.csv"

print("Loading Karnataka rainfall dataset...")

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(
    df["Monitoring Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["Rainfall (MM)"] = pd.to_numeric(
    df["Rainfall (MM)"],
    errors="coerce"
)

df = df.dropna(
    subset=["Station Code", "District", "Date", "Rainfall (MM)"]
)

df = df.sort_values(["Station Code", "Date"])

# Build features separately for every station
results = []

for station, group in df.groupby("Station Code"):

    group = group.set_index("Date").sort_index()

    # Insert missing calendar dates as NaN, NOT zero
    group = group[~group.index.duplicated(keep="first")]

    daily = group[
        ["District", "Rainfall (MM)"]
    ].asfreq("D")

    rain = daily["Rainfall (MM)"]

    features = pd.DataFrame(index=daily.index)

    features["District"] = daily["District"]

    features["Rainfall_Lag_1"] = rain.shift(1)
    features["Rainfall_Lag_2"] = rain.shift(2)
    features["Rainfall_Lag_3"] = rain.shift(3)
    features["Rainfall_Lag_7"] = rain.shift(7)

    previous_rain = rain.shift(1)

    features["Rolling_Mean_3"] = (
        previous_rain.rolling(3, min_periods=3).mean()
    )

    features["Rolling_Sum_7"] = (
        previous_rain.rolling(7, min_periods=7).sum()
    )

    features["Month"] = features.index.month

    features["Day_Of_Year"] = features.index.day_of_year

    # Tomorrow's actual rainfall
    features["Rainfall_Tomorrow"] = rain.shift(-1)

    features["Rain_Tomorrow"] = (
        features["Rainfall_Tomorrow"] > 0
    ).astype(float)

    # Keep only rows with complete historical features
    # and an observed next-day target
    features = features.dropna()

    results.append(features)

print("Combining station features...")

final_df = pd.concat(results)

final_df = final_df.reset_index(names="Date")

final_df["Rain_Tomorrow"] = (
    final_df["Rain_Tomorrow"].astype(int)
)

final_df.to_csv(OUTPUT, index=False)

print("\nFeature preparation completed!")
print("Total ML records:", len(final_df))
print("Rain tomorrow:", final_df["Rain_Tomorrow"].sum())
print(
    "No rain tomorrow:",
    (final_df["Rain_Tomorrow"] == 0).sum()
)

print("\nOutput saved:", OUTPUT)
print("\nFeature columns:")
print(final_df.columns.tolist())

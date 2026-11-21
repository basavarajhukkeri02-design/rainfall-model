
import pandas as pd

df = pd.read_csv("karnataka_rainfall_data_2023_2025.csv")

df["Date"] = pd.to_datetime(
    df["Monitoring Date"],
    format="%d-%m-%Y"
)

coverage = []

for station, group in df.groupby("Station Code"):

    dates = group["Date"].drop_duplicates()

    start = dates.min()
    end = dates.max()

    expected = (end - start).days + 1
    actual = len(dates)

    percentage = (actual / expected) * 100

    coverage.append({
        "Station": station,
        "Start Date": start,
        "End Date": end,
        "Recorded Days": actual,
        "Expected Days": expected,
        "Coverage %": round(percentage, 2)
    })

result = pd.DataFrame(coverage)

print("Total stations:", len(result))

print("\nCoverage summary:")
print(result["Coverage %"].describe())

print("\nLowest 15 coverage stations:")
print(
    result.sort_values("Coverage %").head(15).to_string(index=False)
)

print("\nStations with 95% or higher coverage:",
      (result["Coverage %"] >= 95).sum())

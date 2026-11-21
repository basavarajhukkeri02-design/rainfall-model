
import pandas as pd

df = pd.read_csv("karnataka_rainfall_data_2023_2025.csv")

df["Date"] = pd.to_datetime(
    df["Monitoring Date"],
    format="%d-%m-%Y"
)

# Check duplicate station-date records
duplicates = df.duplicated(
    subset=["Station Code", "Date"]
)

print("Duplicate station-date records:", duplicates.sum())

# Check missing dates for each station
gap_results = []

for station, group in df.groupby("Station Code"):

    dates = group["Date"].sort_values().drop_duplicates()

    if len(dates) > 1:
        expected_days = (dates.max() - dates.min()).days + 1
        missing_days = expected_days - len(dates)

        gap_results.append({
            "Station": station,
            "Records": len(dates),
            "Missing days": missing_days
        })

gaps = pd.DataFrame(gap_results)

print("\nStations with missing dates:")
print((gaps["Missing days"] > 0).sum())

print("\nTop 15 stations with most missing dates:")
print(
    gaps.sort_values(
        "Missing days",
        ascending=False
    ).head(15).to_string(index=False)
)


import pandas as pd

df = pd.read_csv("karnataka_rainfall_data_2023_2025.csv")

df["Date"] = pd.to_datetime(
    df["Monitoring Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

top = df.nlargest(5, "Rainfall (MM)")

for _, record in top.iterrows():
    station = record["Station Code"]
    date = record["Date"]

    print("\n" + "=" * 60)
    print("Station:", station)
    print("District:", record["District"])
    print("Extreme rainfall:", record["Rainfall (MM)"], "mm")
    print("Date:", date)

    nearby = df[
        (df["Station Code"] == station) &
        (df["Date"].between(
            date - pd.Timedelta(days=3),
            date + pd.Timedelta(days=3)
        ))
    ]

    print("\nRainfall around this date:")
    print(
        nearby[["Date", "Rainfall (MM)"]]
        .sort_values("Date")
        .to_string(index=False)
    )

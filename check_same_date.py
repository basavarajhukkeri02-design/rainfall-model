
import pandas as pd

df = pd.read_csv("karnataka_rainfall_data_2023_2025.csv")

df["Date"] = pd.to_datetime(
    df["Monitoring Date"],
    format="%d-%m-%Y"
)

dates = [
    "2024-08-05",
    "2024-07-24",
    "2023-09-16",
    "2024-06-28",
    "2024-07-11"
]

for date in dates:
    print("\n==============================")
    print("DATE:", date)

    selected = df[df["Date"] == date]

    result = selected.groupby("District")["Rainfall (MM)"].agg(
        ["count", "median", "max"]
    )

    print(result.sort_values("max", ascending=False).head(8))

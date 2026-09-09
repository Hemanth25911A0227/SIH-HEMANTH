"""
Phase 1a — intake. Normalizes your BigQuery daily export + climatology table
into data/raw/. Usage:
    python src/fetch_data.py --bq bq-results-...csv --clim climatological.csv
"""
import argparse
import pandas as pd
from preprocess import RAW, SENSORS

NEED = ["Station_ID", "Station_Name", "Latitude", "Longitude",
        "Elevation_m", "Timestamp"] + SENSORS

def load_bq(path: str, station: str = "SRINAGAR") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    missing = [c for c in NEED if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns {missing}; found {list(df.columns)}")
    if df["Station_Name"].nunique() > 1:
        hit = df[df["Station_Name"].str.upper().str.strip() == station.upper().strip()]
        if len(hit) > 0:
            print(f"multiple stations found — filtering to {station} ({len(hit)} rows)")
            df = hit
        else:
            top = df["Station_ID"].value_counts().idxmax()
            print(f"station '{station}' not found — falling back to Station_ID {top}")
            df = df[df["Station_ID"] == top]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return (df.sort_values("Timestamp").drop_duplicates("Timestamp")
              .reset_index(drop=True))[NEED]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bq", required=True)
    ap.add_argument("--clim", default=None)
    a = ap.parse_args()

    df = load_bq(a.bq)
    df.to_csv(RAW / "srinagar_daily.csv", index=False)
    print(f"station : {df['Station_Name'].iloc[0]} (id {df['Station_ID'].iloc[0]}, "
          f"elev {df['Elevation_m'].iloc[0]} m)")
    print(f"rows    : {len(df)} ({df.Timestamp.min().date()} -> {df.Timestamp.max().date()})")
    print(f"NaNs    : {int(df[SENSORS].isna().sum().sum())}")

    if a.clim:
        clim = pd.read_csv(a.clim)
        clim.to_csv(RAW / "climatological.csv", index=False)
        hit = clim["station_name"].astype(str).str.upper().str.strip() \
              == str(df["Station_Name"].iloc[0]).upper().strip()
        print(f"climatology rows: {len(clim)}; station found in it: {bool(hit.any())}")

if __name__ == "__main__":
    main()

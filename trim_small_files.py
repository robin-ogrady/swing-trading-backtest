"""
trim_small_files.py
Deletes every data file with fewer than N rows (default 500) and removes it from data/index.csv.

  python3 trim_small_files.py --dry-run     (just shows what would be deleted)
  python3 trim_small_files.py               (deletes files under 500 rows)
  python3 trim_small_files.py --min-rows 1000
Run it from the folder that contains data/. It reads row counts from data/index.csv,
so it doesn't need pyarrow.
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--min-rows", type=int, default=500)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

data = Path("data")
idx_path = data / "index.csv"
idx = pd.read_csv(idx_path)
small = idx[idx.rows < a.min_rows]

print(f"{len(idx)} files listed, {len(small)} have fewer than {a.min_rows} rows")
print(small.groupby(["asset_class", "timeframe"]).size().to_string() if len(small) else "nothing to remove")

if a.dry_run or small.empty:
    raise SystemExit("dry run, nothing deleted" if a.dry_run else 0)

removed = 0
for _, r in small.iterrows():
    f = data / r.asset_class / r.timeframe / f"{r.symbol}.parquet"
    if f.exists():
        f.unlink()
        removed += 1

idx[idx.rows >= a.min_rows].to_csv(idx_path, index=False)
print(f"deleted {removed} files, kept {len(idx) - len(small)}. index.csv updated.")

import numpy as np
import csv
import os
from tqdm import tqdm
from markov import get_migration_time

kh_values = np.logspace(0, 2, 50)   # 0.01 to 100
nm_values = np.arange(1, 21)         # 1 to 20

rows = []
total = len(kh_values) * len(nm_values)

with tqdm(total=total, desc="Sweeping") as pbar:
    for n_med in nm_values:
        for kh in kh_values:
            mean_t = get_migration_time(kh=kh, n_mediators=n_med)
            rows.append((kh, n_med, mean_t))
            pbar.update(1)

out_path = os.path.join(os.path.dirname(__file__), "migration_times2.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kh", "n_mediators", "mean_t"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out_path}")

import pandas as pd
import os

# ── Configuration ──────────────────────────────────────────────────────────────
# Directory containing per-type CSVs produced by compute_avg.py (ST3/MT3/T4)
INPUT_DIR = "/path/to/output/avg/"
# Output directory for sampled CSVs
OUTPUT_DIR = "/path/to/output/sampled/"
# Number of pairs to sample per category
SAMPLE_SIZE = 500
# Random seed for reproducibility (set to None for non-deterministic sampling)
RANDOM_SEED = 42
# ───────────────────────────────────────────────────────────────────────────────

CLONE_TYPES = ['ST3', 'MT3', 'T4']


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ctype in CLONE_TYPES:
        in_path = os.path.join(INPUT_DIR, f"{ctype}.csv")
        if not os.path.exists(in_path):
            print(f"[{ctype}] file not found: {in_path}, skipping.")
            continue

        df = pd.read_csv(in_path)
        total = len(df)

        if total <= SAMPLE_SIZE:
            sample = df
            print(f"[{ctype}] only {total} pairs available, using all.")
        else:
            sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
            print(f"[{ctype}] sampled {SAMPLE_SIZE} from {total} pairs.")

        out_path = os.path.join(OUTPUT_DIR, f"{ctype}_500.csv")
        sample.to_csv(out_path, index=False)
        print(f"[{ctype}] saved -> {out_path}")


if __name__ == '__main__':
    main()

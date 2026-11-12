import pandas as pd
import os

# ── Configuration ──────────────────────────────────────────────────────────────
# Path to similarity CSV produced by main.py (clone pairs: F1, F2, t1_sim, t2_sim, t3_sim)
CLONE_SIM_CSV = "/path/to/output/clone_sim.csv"
# Output directory for per-type averaged CSVs
OUTPUT_DIR = "/path/to/output/avg/"
# ───────────────────────────────────────────────────────────────────────────────

# Clone type classification based on avg_sim:
#   ST3 : avg_sim >= 0.7
#   MT3 : 0.5 <= avg_sim < 0.7
#   T4  : avg_sim < 0.5


def classify(avg_sim):
    if avg_sim >= 0.7:
        return 'ST3'
    elif avg_sim >= 0.5:
        return 'MT3'
    else:
        return 'T4'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CLONE_SIM_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Compute average similarity
    df['t1_sim'] = pd.to_numeric(df['t1_sim'], errors='coerce')
    df['t2_sim'] = pd.to_numeric(df['t2_sim'], errors='coerce')
    df['t3_sim'] = pd.to_numeric(df['t3_sim'], errors='coerce')
    df['avg_sim'] = df[['t1_sim', 't2_sim', 't3_sim']].mean(axis=1)

    # Classify by avg_sim
    df['Type'] = df['avg_sim'].apply(classify)

    # Save per-type CSVs
    for ctype in ['T4', 'ST3', 'MT3']:
        subset = df[df['Type'] == ctype]
        if subset.empty:
            print(f"[{ctype}] no data, skipping.")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{ctype}.csv")
        subset.to_csv(out_path, index=False)
        print(f"[{ctype}] {len(subset)} pairs -> {out_path}")


if __name__ == '__main__':
    main()

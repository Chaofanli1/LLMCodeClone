# Token Selection

This directory contains scripts for constructing the GCJ clone detection dataset using three token-based similarity tools.

---

## Scripts

| Script | Description |
|--------|-------------|
| `runner.py` | Unified dispatcher; calls the corresponding tool by tool code |
| `main.py` | Batch similarity computation for clone pairs (multiprocessing, 60 cores) |
| `compute_avg.py` | Computes the average of t1/t2/t3 similarity scores and splits pairs by clone type |
| `sample_pairs.py` | Randomly samples 500 pairs from each clone type category |

---

## Dataset Construction

The dataset was constructed using the following procedure:

1. **Compute similarities.** For each code pair, run all three token-based tools (NiCad, SourcererCC, LVMapper) to obtain similarity scores `t1_sim`, `t2_sim`, `t3_sim`.

2. **Average the scores.** Compute the mean of the three similarity scores for each pair:

   ```
   avg_sim = (t1_sim + t2_sim + t3_sim) / 3
   ```

3. **Classify by avg_sim.** Each pair is assigned a clone type based on its average similarity score:
   - ST3 : avg_sim ≥ 0.7
   - MT3 : 0.5 ≤ avg_sim < 0.7
   - T4  : avg_sim < 0.5

4. **Sample 500 pairs per category.** From each category, randomly sample 500 pairs to form a balanced dataset for downstream experiments.

---

## Data Flow

```
Input:
  clone_pairs.csv        — code pairs (F1, F2) from GCJ

Step 1 — Compute similarities:
  main.py                — computes t1/t2/t3 similarities for each pair

Step 2 — Average & classify:
  compute_avg.py         — computes avg_sim = mean(t1_sim, t2_sim, t3_sim)
                           assigns clone type label by avg_sim threshold:
                             ST3: avg_sim >= 0.7
                             MT3: 0.5 <= avg_sim < 0.7
                             T4:  avg_sim < 0.5

Step 3 — Sample:
  sample_pairs.py        — randomly samples 500 pairs from each clone type

Output:
  avg/<type>.csv         — all clone pairs per type with avg_sim
  sampled/<type>_500.csv — 500 sampled pairs per type
```

---

## Dependencies

- Python 3
- `javalang`: Java parser
- `numpy`, `pandas`

---

## Notes

1. Set `inputpath` in `main.py` to the directory containing your `.java` function files.
2. Set `INPUT_CSV` and `OUTPUT_DIR` in `main.py` to match your environment.
3. Set `CLONE_SIM_CSV` and `OUTPUT_DIR` in `compute_avg.py` accordingly.
4. Set `INPUT_DIR` and `OUTPUT_DIR` in `sample_pairs.py` accordingly.

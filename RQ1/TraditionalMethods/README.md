# Traditional Code Clone Detection Tools

This directory contains implementation scripts for the 9 traditional code clone detection tools used in the paper experiments.

---

## Tool List

### Token-based

| Script | Tool | Code | Threshold |
|--------|------|------|-----------|
| `Token_NiCad.py` | NiCad | t1 | 0.7 |
| `Token_SourcererCC.py` | SourcererCC | t2 | 0.7 |
| `Token_LVMapper.py` | LVMapper | t3 | 0.7 |

### AST-based

| Script | Tool | Code | Threshold |
|--------|------|------|-----------|
| `AST_AST2014.py` | Lazar (AST2014) | a1 | 0.8 |
| `AST_BWCCA2015.py` | Zhao (BWCCA2015) | a2 | 0.65 |
| `AST_COMPSAC2018.py` | Yang (COMPSAC2018) | a3 | 0.8 |

### CFG-based

| Script | Tool | Code | Threshold |
|--------|------|------|-----------|
| `CFG_StoneDetector.py` | StoneDetector | c1 | 0.8 |
| `CFG_Centroids.py` | GroupDroid (Centroids) | c2 | 0.95 |
| `CFG_ATVHunter2.py` | ATVHunter | c3 | 0.85 |

---

## Dispatcher Scripts

| Script | Description |
|--------|-------------|
| `runner.py` | Unified dispatcher; calls the corresponding tool by tool code |
| `main.py` | Batch similarity computation for clone pairs (multiprocessing, 60 cores) |
| `call.py` | Batch similarity computation for non-clone pairs (multiprocessing, 30 cores) |
| `detectpath.py` | Path configuration helper |

---

## Data Flow

```
Input:
  Clone pairs
  Non-clone pairs

Function files:
  GCJ
  BCB
  CFG dot files

Output:
  Clone pair similarities
  Non-clone pair similarities
  Recall statistics
```

---

## Dependencies

- Python 3
- `javalang`: Java parser for Token/AST tools
- `anytree`: AST tools
- `networkx`, `pydot`: CFG tools
- `numpy`, `pandas`

---

## Notes

1. **CFG tools** require pre-generated `.dot` files (produced by Joern from Java source), and cannot run directly on function strings.
2. Before running CFG tools, set `JOERN_PATH` in `detectpath.py`.
3. Input/output paths in `main.py` and `call.py` must be updated to match your environment.
4. `CFG_StoneDetector.py` and `CFG_Centroids.py` include pydot/networkx compatibility fixes.

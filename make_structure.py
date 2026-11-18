
#!/usr/bin/env python3
"""Scaffold the project: folders, stub source files, README, requirements,
and a build helper for the LaTeX report.

Usage:
    python make_structure.py [project_dir]
The LaTeX template (report.tex) is separate: copy it into <project_dir>/report/ yourself.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "ai651_pa3")

dirs = ["data/raw", "data/processed", "src", "notebooks",
        "results", "report/figures", "checkpoints"]
for d in dirs:
    (root / d).mkdir(parents=True, exist_ok=True)

files = {
"README.md": """# AI651 Assignment 3: Spatio-Temporal Traffic Forecasting (LargeST-SD)

## Setup
pip install -r requirements.txt

## Data
Download LargeST from Kaggle and place the SD files in data/raw/.
python src/data.py          # resample, split, normalise -> data/processed/

## Run experiments (seed 42)
python src/train.py --model temporal
python src/train.py --model fixed
python src/train.py --model adaptive
python src/train.py --model hybrid

## Figures and report
python src/plots.py         # writes PDFs to report/figures/
cd report && latexmk -pdf report.tex
""",
"requirements.txt": "numpy\npandas\ntables\nh5py\ntorch\nmatplotlib\nscipy\ntqdm\n",
".gitignore": "data/raw/\ndata/processed/\ncheckpoints/\n__pycache__/\n*.aux\n*.log\n*.out\n*.toc\n*.fls\n*.fdb_latexmk\n*.synctex.gz\n",
"src/__init__.py": "",
"src/utils.py": '''import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
''',
"src/data.py": '''"""Q3/Q4(a): load SD subset, resample to 15 min (sum), handle NaNs,
chronological splits, train-only normalisation, window construction."""

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
''',
"src/model.py": '''"""Q4/Q5: temporal encoder, graph conv, head, adjacency providers
(fixed / adaptive / hybrid)."""
import torch
import torch.nn as nn
''',
"src/train.py": '''"""Training and evaluation. Usage: python src/train.py --model {temporal,fixed,adaptive,hybrid}"""
import argparse
from utils import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["temporal", "fixed", "adaptive", "hybrid"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    raise NotImplementedError

if __name__ == "__main__":
    main()
''',
"src/plots.py": '''"""Generate report figures into report/figures/ (PDF):
q2a_heatmaps, q3a_sensors_week, q3b_adj_heatmap, q3b_degree_hist,
q5b_adaptive_adj, q6a_pred_vs_actual."""
''',
"report/build.sh": """#!/usr/bin/env bash
cd "$(dirname "$0")" && latexmk -pdf -interaction=nonstopmode report.tex
""",
}

for name, content in files.items():
    path = root / name
    if not path.exists():
        path.write_text(content)

(root / "report/build.sh").chmod(0o755)

print(f"Created project at: {root.resolve()}")
for p in sorted(root.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(root))

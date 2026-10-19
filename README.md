# What Helped with Huge Data Hurt with Small Data
Code & data for [*"What Helped with Huge Data Hurt with Small Data: Standardization, Representational Geometry, and Data Scale."*](WhatHelpedWithHugeDataHurtWithSmallData.pdf)


## Current Release
Version 2 adds the code that generated the results from the models. The repository contains:

- `data/`: the per-model, per-layer results (see `data/README.md`).
- `notebooks/paper_analyses.ipynb`: reproduces every analysis, table, and figure in the paper from the results in `data/`.
- `generation/`: the code that produced `data/` from the models, with configs for every model and checkpoint in the paper (see `generation/README.md`).
- `data/inputs/`: the Wikipedia context sentences and corpus the generation code uses.

To reproduce the paper's analyses, you only need `data/` and the notebook. Use `generation/` to regenerate results or to apply the same measurements to other models.

## How to Run
Requires Python 3.11 or newer (tested with 3.14).

Create and activate a virtual environment, then install the requirements:
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use `notebooks/paper_analyses.ipynb` to reproduce the analyses reported in the paper; open in VS Code or another Jupyter-compatible editor.

To regenerate results from the models, see `generation/README.md`. Its dependencies (including PyTorch) are in the same `requirements.txt`.

## Notes on the paper text

Two small inconsistencies between the published paper and the code, neither of which affects any conclusion:

- **Section 4.1.1, huge-data mean.** The paper reports mean Δρ = 0.0369 for the four huge-data models. That is the mean over all layers of all four models. The mean of the four per-model averages, which is what the reported p-value (0.015) and Figure 1 use, is 0.0395. The CP-data mean (−0.0158) is a per-model average.
- **Section 4.1.2, the 100B boundary.** The paper describes the huge-data checkpoints as those trained on more than 100B tokens. The code uses 100B or more, which includes the Gemstone-256x23 and Gemstone-512x11 checkpoints at step 23850 (exactly 100B tokens). Table 2 reflects the code.


## Citation
If you use this code or data, please cite:
```bibtex
@inproceedings{finegan-dollak-2026-standardization,
  title     = {What Helped with Huge Data Hurt with Small Data: Standardization, Representational Geometry, and Data Scale},
  author    = {Finegan-Dollak, Catherine},
  booktitle = {Proceedings of the Fourth BabyLM Workshop: Accelerating Language Modeling Research with Cognitively Plausible Datasets},
  year      = {2026}
}
```

## Reproducibility note

The mediation analyses in `notebooks/paper_analyses.ipynb` use Pingouin's nonparametric bootstrap, so their p-values and confidence intervals vary slightly from run to run. Runs reported in the paper did not set a random seed, and most mediation cells in this notebook are not seeded either, so values from rerunning the notebook may differ slightly from the published ones. All other analyses are deterministic.

These differences do not change the paper's conclusions. Indirect effects that were clearly significant or clearly non-significant in the paper remain so across runs. A few p-values lie close to 0.05 and may fall on either side of that threshold on a given run; the paper's main findings (the reversal in standardization effect, the correlation of rogue dimensionality and anisotropy with data scale, and the absence of general positive mediation by either) do not depend on any single near-threshold result.

## License

Code in this repository is released under the MIT License (see LICENSE). Data files in data/ are released under CC BY 4.0 (see data/README.md), except the Wikipedia-derived files in data/inputs/, which are under CC BY-SA 4.0 (see data/inputs/README.md).
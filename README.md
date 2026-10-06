# What Helped with Huge Data Hurt with Small Data
Code & data for [*"What Helped with Huge Data Hurt with Small Data: Standardization, Representational Geometry, and Data Scale."*](WhatHelpedWithHugeDataHurtWithSmallData.pdf)


## Current Release
This release contains the per-model, per-layer results (in `data/`) and the notebook that reproduces every analysis, table, and figure in the paper from those results. Code for generating the results files from the models is not included in this release.

## How to Run
Requires Python 3.11 or newer (tested with 3.14).

Create and activate a virtual environment, then install the requirements:
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use `notebooks/paper_analyses.ipynb` to reproduce the analyses reported in the paper; open in VS Code or another Jupyter-compatible editor.


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

Code in this repository is released under the MIT License (see LICENSE). Data files in data/ are released under CC BY 4.0 (see data/README.md).
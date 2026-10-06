# Data

Per-model, per-layer results for *"What Helped with Huge Data Hurt with Small Data: Standardization, Representational Geometry, and Data Scale."* Each CSV has one row per model, checkpoint, and layer (and per condition, where noted). See the paper for full details of how each measure was computed.

| File | Rows | Contents |
|---|---|---|
| `simlex_results.csv` | 17,148 | SimLex-999 performance with and without standardization |
| `anisotropy.csv` | 8,567 | Anisotropy of each layer |
| `dim_reduction.csv` | 25,701 | Rogue dimensionality of each layer, for k = 1, 3, 5 |
| `isolated_vs_contextualized.csv` | 352 | Appendix A comparison of isolated and contextualized word vectors |

## Models and checkpoints

The data cover 18 models:

- **CP-scale models (one checkpoint each):** BabyLlama, OPT-125M, Babylm-Baseline GPT2, and GPT-BERT (Causal-Focus, Masked-Focus, Mixed), each in a 10M-word and a 100M-word version; and BaselineLM, trained on 10M words of either LocNar+Wiki or The Pile.
- **Checkpoint-sequence models:** Pythia-14M, Pythia-70M, Gemstone-256x23, and Gemstone-512x11, with results for every published checkpoint.

## Shared columns

These columns appear in `simlex_results.csv`, `anisotropy.csv`, and `dim_reduction.csv`.

| Column | Description |
|---|---|
| `name` | Model name. |
| `layer` | Layer index. Layer 0 is the word embedding layer, so a model with *n* transformer layers has layers 0 through *n*. |
| `step` | Training step of the checkpoint. Blank for the fully trained model, except for Gemstone, whose final checkpoint is step 82998. |
| `data_size` | Training data scale of the checkpoint: `10M` or `100M` (words), `Huge` (the final checkpoint of a Pythia or Gemstone model, about 300B and 350B tokens respectively), or `Unknown` (an intermediate Pythia or Gemstone checkpoint; use `step` or `tokens_seen` instead). |

The "across models" analyses in the paper use the rows whose `data_size` is `10M`, `100M`, or `Huge`. The "within models" analyses use the rows with a non-blank `step`.

## `simlex_results.csv`

Two rows per model, checkpoint, and layer: one for each value of `standardization`.

| Column | Description |
|---|---|
| `spearman` | Spearman's ρ between the cosine similarity of the two words' vectors and the human similarity rating, over the SimLex-999 word pairs. Word vectors are contextualized: the mean over ten Wikipedia sentences containing the word, with subword vectors mean-pooled. |
| `standardization` | `Standardized` if the vectors were standardized (mean subtracted, divided by standard deviation) before computing cosine similarity; otherwise `Not Standardized`. |

This file has no `tokens_seen` column; the notebook computes it from `step`.

## `anisotropy.csv`

One row per model, checkpoint, and layer.

| Column | Description |
|---|---|
| `anisotropy` | Average cosine similarity of 500k random pairs of tokens drawn from Wikipedia sentences. Values near 1 indicate high anisotropy; values near 0 indicate high isotropy. |
| `tokens_seen` | Number of training tokens seen at this checkpoint. Filled in only where `step` is: `step` × 2,097,152 for Pythia, and `step` × 2,000,000,000 / 477 for Gemstone. |

## `dim_reduction.csv`

Three rows per model, checkpoint, and layer: one for each value of `k`.

| Column | Description |
|---|---|
| `k` | Number of top dimensions removed, as the string `k=1`, `k=3`, or `k=5`. |
| `r2` | Squared Pearson correlation between the cosine similarity of token pairs using all dimensions and their cosine similarity with the top `k` dimensions removed. Values near 1 mean those dimensions contribute little; lower values mean greater rogue dimensionality. |
| `tokens_seen` | As in `anisotropy.csv`. |

## `isolated_vs_contextualized.csv`

Results for the fully trained versions of eight models (BaselineLM, BabyLlama, OPT-125M, and Pythia). Four rows per model and layer: one for each combination of `embedding_mode` and `standardization`. Some columns are named differently from the other files.

| Column | Description |
|---|---|
| `name` | Model name, as in the other files. |
| `model_cls` | Hugging Face model ID, or the internal class name for BaselineLM. |
| `embedding_mode` | `isolated` if each word was passed through the model with no context; `contextualized` if its vector is the mean over ten Wikipedia sentences containing it. |
| `spearman` | As in `simlex_results.csv`. |
| `layer_num` | Layer index; same meaning as `layer` in the other files. |
| `standardization` | As in `simlex_results.csv`. |
| `Data Scale` | `10M`, `100M`, or `Huge`; same meaning as `data_size` in the other files. |

## Known quirks

- `simlex_results.csv` contains each Pythia-70M step 70000 row twice (14 exact duplicate rows). However, every analysis that uses simlex_results.csv goes through compute_delta_spearman, which calls pivot_table, so the 14 duplicated rows collapse back into one before Δρ is ever computed.
- Pythia-70M has no step 14000 checkpoint in any file.
- Pythia's last numbered checkpoint (step 143000) is included with `data_size` `Unknown`, separately from the fully trained model (blank `step`, `data_size` `Huge`). These are likely the same weights. Within-model analyses use only the step 143000 rows and across-model analyses use only the fully trained rows, so no analysis counts them twice.

## License

The data files in this directory are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/

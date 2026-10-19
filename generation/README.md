# Generating the results

This folder contains the code that produced the CSVs in `data/` from the models. It is provided as-is: it is the code used for the paper, lightly cleaned for release (paths made relative, unreleased model code removed). It has been verified on Pythia-14M (see "Smoke test" below), not on every model.

All commands below are run from the repository root.

## What each script does

| Script | Output | Feeds into |
|---|---|---|
| `gather_context_sentences.py` | `data/inputs/simlex_context.json` | SimLex scoring |
| `gather_corpus.py` | `data/inputs/corpus.txt` | Standardization and anisotropy |
| `simlex_comparison.py` | `results/simlex/<timestamp>/summary.csv` | `simlex_results.csv`, `isolated_vs_contextualized.csv` |
| `anisotropy.py` | `<output_dir>/anisotropy.csv`, `dim_reduction_r2.csv`, `per_dim_contribution.csv` | `anisotropy.csv`, `dim_reduction.csv` |
| `assemble_csvs.py` | The four CSVs in `data/` | The analysis notebook |
| `compare_to_release.py` | A printed report | Checking new results against `data/` |

Supporting modules: `run_simlex.py` (SimLex loading and per-layer scoring), `similarity_scoring.py` (word embedding and standardization), and `hf_wrapper.py` (gives Hugging Face models a common interface).

You do not need to run the two `gather_` scripts: their outputs are released in `data/inputs/` and are the canonical inputs. The scripts are included to document how those files were made.

## Inputs

- `data/inputs/simlex_context.json` and `data/inputs/corpus.txt` are included. See `data/inputs/README.md` for their license.
- SimLex-999 is not included. Download it from the [SimLex-999 page](https://fh295.github.io/simlex.html), unzip it, and copy `SimLex-999.txt` to `data/inputs/SimLex-999.txt`.

## Method details

These details are implemented in the code and are useful for anyone extending the work.

- **Context sentences.** For each SimLex-999 word, a pool of 50 candidate sentences is collected from the English Wikipedia dump (`wikimedia/wikipedia`, config `20231101.en`), keeping sentences of 6 to 75 words and excluding sentences in the BaselineLM's WIT training data. Ten are then sampled (seed 42), preferring sentences where the word appears in lowercase (so not as a proper noun). A word's vector at a layer is the mean over its sentences of the mean of its subword vectors.
- **Standardization.** The per-layer mean and standard deviation are computed per dimension over every token (including any special tokens the tokenizer adds) of every sentence in `corpus.txt`, using the population standard deviation. In the configs, `"corpus_sizes": [null, "all"]` produces both conditions: `null` is Not Standardized and `"all"` is Standardized.
- **Corpus.** `corpus.txt` is about 500,000 words of English Wikipedia (same dump), excluding sentences in the BaselineLM's WIT training data. It is the same corpus for standardization and anisotropy. Note that only WIT sentences are excluded; no other datasets were checked for overlap.
- **Anisotropy and rogue dimensionality.** 500,000 random token pairs are sampled (seed 42) from `corpus.txt`. Anisotropy is their mean cosine similarity. For rogue dimensionality, dimensions are ranked by their mean contribution to cosine similarity, and R² is computed between the full cosine similarity and the cosine similarity with the top k dimensions removed, for k = 1, 3, 5.

## Smoke test (Pythia-14M)

This runs both measurements on the fully trained Pythia-14M and checks the output against the released rows. Runs on a laptop CPU are expected to take minutes; the anisotropy step holds hidden states for the whole corpus in memory (a few GB for Pythia-14M).

First set up the environment as described in the top-level `README.md` (`pip install -r requirements.txt`), then:

```
python generation/simlex_comparison.py --config generation/configs/smoke_test.json
python generation/anisotropy.py --config generation/configs/smoke_test.json

python generation/compare_to_release.py \
    --simlex-summary results/simlex/*/summary.csv \
    --anisotropy results/anisotropy_smoke_test/anisotropy.csv \
    --dim-reduction results/anisotropy_smoke_test/dim_reduction_r2.csv
```

The last command prints `MATCH` or `MISMATCH` for each measure. `anisotropy.py` appends to its output files, so delete `results/anisotropy_smoke_test/` before re-running it. Each `simlex_comparison.py` run adds a new timestamped folder under `results/simlex/`, so if earlier runs are there, replace the `results/simlex/*/summary.csv` glob with the path to the newest run's `summary.csv`.

## Full reproduction

The configs in `configs/` list every model and checkpoint in the paper except the two BaselineLM models (see [Not included](#not-included)), rebuilt from the model IDs and revisions recorded in the original raw outputs:

| Config | Contents |
|---|---|
| `final_models.json` | The 12 BabyLM-scale Hugging Face models and fully trained Pythia-14M and Pythia-70M |
| `pythia_14m_checkpoints.json`, `pythia_70m_checkpoints.json` | Every Pythia checkpoint used |
| `gemstone_256x23_checkpoints.json`, `gemstone_512x11_checkpoints.json` | Every Gemstone checkpoint used, including the final one |
| `isolated_embeddings.json` | Appendix A isolated-embedding runs (pair with the contextualized runs of the same models) |
| `smoke_test.json` | Fully trained Pythia-14M only |

Each config works for both `simlex_comparison.py` and `anisotropy.py`. The paper's runs used GPUs on a cluster; together these configs cover 666 model checkpoints. `simlex_comparison.py` can split work across processes with `torchrun` or SLURM.

Then combine the raw outputs:

```
python generation/assemble_csvs.py \
    --simlex-summaries results/simlex/*/summary.csv \
    --anisotropy results/anisotropy/anisotropy.csv \
    --dim-reduction results/anisotropy/dim_reduction_r2.csv \
    --out-dir results/assembled
```

Run on the original raw outputs, `assemble_csvs.py` reproduces `simlex_results.csv`, `anisotropy.csv`, and `dim_reduction.csv` exactly, and `isolated_vs_contextualized.csv` with the same rows in a different order. (Keep the isolated-embedding summaries out of `--simlex-summaries`; pass them, together with the matching contextualized summaries, to `--isolated-summaries`.)

## Not included

The two BaselineLM models were trained for this paper and their weights are not currently released, so their rows cannot be regenerated with this code. The scripts accept only pretrained Hugging Face models.

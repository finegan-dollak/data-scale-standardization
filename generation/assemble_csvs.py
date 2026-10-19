"""Combine raw outputs from simlex_comparison.py and anisotropy.py into the CSVs in data/.

This is the step between the raw per-run outputs and the four released CSVs
(simlex_results.csv, anisotropy.csv, dim_reduction.csv,
isolated_vs_contextualized.csv). It is a script version of the notebook cells
that built the v1 CSVs; run on the original raw outputs, it reproduces the
released files exactly (isolated_vs_contextualized.csv has the same rows in a
different order).

Usage (from the repository root):

    python generation/assemble_csvs.py \
        --simlex-summaries results/simlex/*/summary.csv \
        --anisotropy results/anisotropy/anisotropy.csv \
        --dim-reduction results/anisotropy/dim_reduction_r2.csv \
        --isolated-summaries results/simlex_isolated/*/summary.csv \
        --out-dir results/assembled

Each flag takes one or more files; they are concatenated in the order given.
--isolated-summaries is optional. When given, it should list the summaries of
the isolated-embedding runs AND the contextualized runs of the same models.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_NAMES = {
    "run_20260313_145656 (baselinelmmodel)": "BaselineLM - LocNar+Wiki",
    "pythia-14m": "Pythia-14M",
    "pythia-70m": "Pythia-70M",
    "babyllama": "BabyLlama-10M",
    "babyllama-100m": "BabyLlama-100M",
    "opt-125m-strict-small-2023": "OPT-125M-10M",
    "opt-125m-strict-2023": "OPT-125M-100M",
    "the pile (baselinelmmodel)": "BaselineLM - The Pile",
    "babylm-baseline-100m-gpt-bert-causal-focus": "GPT-BERT (Causal-Focus) - 100M",
    "babylm-baseline-100m-gpt-bert-mixed": "GPT-BERT (Mixed) - 100M",
    "babylm-baseline-10m-gpt-bert-mixed": "GPT-BERT (Mixed) - 10M",
    "babylm-baseline-10m-gpt-bert-causal-focus": "GPT-BERT (Causal-Focus) - 10M",
    "babylm-baseline-10m-gpt-bert-masked-focus": "GPT-BERT (Masked-Focus) - 10M",
    "babylm-baseline-100m-gpt-bert-masked-focus": "GPT-BERT (Masked-Focus) - 100M",
    "babylm-baseline-100m-gpt2": "Babylm-Baseline GPT2 - 100M",
    "babylm-baseline-10m-gpt2": "Babylm-Baseline GPT2 - 10M",
    "gemstone-256x23": "Gemstone-256x23",
    "gemstone-512x11": "Gemstone-512x11",
}

STEP_PATTERN = r"@step_*(\d+)"


def canonicalize_model_name(name):
    """Map run names (with or without '@step...' suffixes) to the names used in data/."""
    key = name.lower()
    if "/" in key:
        key = key.split("/")[-1]
    if "@step" in key:
        key = key.split("@step")[0]
    canonical = CANONICAL_NAMES.get(key)
    if canonical is None:
        print(f"Warning: no canonical name for {name!r}; keeping it as is.")
        return key
    return canonical


def tokens_seen(model, step):
    """Training tokens seen at a checkpoint step, for Pythia and Gemstone only."""
    if pd.isna(step):
        return None
    if "pythia" in model.lower():
        return step * 2_097_152
    if "gemstone" in model.lower():
        return step * 2_000_000_000 // 477
    return None


def add_data_size(df, max_gemstone_step):
    """Label each row's training data scale: 10M, 100M, Huge, or Unknown."""
    df["data_size"] = "Unknown"
    name = df["name"]
    df.loc[name.str.contains("BaselineLM", case=False), "data_size"] = "10M"
    df.loc[name.str.contains("10m", case=False), "data_size"] = "10M"
    df.loc[name.str.contains("small", case=False), "data_size"] = "10M"
    df.loc[name.str.contains("strict", case=False) & ~name.str.contains("small", case=False), "data_size"] = "100M"
    df.loc[name.str.contains("100m", case=False), "data_size"] = "100M"
    df.loc[name.str.contains("Pythia", case=False) & df["step"].isnull(), "data_size"] = "Huge"
    df.loc[name.str.contains("Gemstone") & (df["step"] == max_gemstone_step), "data_size"] = "Huge"
    return df


def load_simlex_summaries(paths, exclude=None):
    """Concatenate simlex_comparison.py summaries and keep single-layer rows."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if exclude:
        df = df[~df["model_cls"].str.contains(exclude)]
    df = df[df["layer"].str.startswith("layer_")].copy()
    df["layer"] = df["layer"].str.extract(r"layer_(\d+)").astype(int)
    df["step"] = df["checkpoint"].str.extract(STEP_PATTERN, expand=False).astype("Int64")
    return df.drop_duplicates()


def standardization_label(n_corpus_sentences):
    """'all' corpus sentences = Standardized; no corpus = Not Standardized."""
    return (n_corpus_sentences.astype(str) == "all").map({True: "Standardized", False: "Not Standardized"})


def build_simlex_results(simlex_paths, exclude=None):
    df = load_simlex_summaries(simlex_paths, exclude)
    df = df.drop(columns=["results_path", "embedding_mode", "model_cls", "checkpoint"])
    df["name"] = df["name"].apply(canonicalize_model_name)
    df["standardization"] = standardization_label(df["n_corpus_sentences"])
    return df.drop(columns=["n_corpus_sentences"])


def build_geometry(anisotropy_paths, dim_reduction_paths, exclude=None):
    anisotropy_df = pd.concat([pd.read_csv(p) for p in anisotropy_paths], ignore_index=True)
    dim_reduction_df = pd.concat([pd.read_csv(p) for p in dim_reduction_paths], ignore_index=True)
    out = []
    for df in (anisotropy_df, dim_reduction_df):
        if exclude:
            df = df[~df["run"].str.contains(exclude)]
        df = df.copy()
        df["step"] = df["run"].str.extract(STEP_PATTERN, expand=False).astype("Int64")
        df = df.drop_duplicates().reset_index(drop=True)
        df["name"] = df["run"].apply(canonicalize_model_name)
        df["tokens_seen"] = df.apply(lambda row: tokens_seen(row["name"], row["step"]), axis=1)
        out.append(df.drop(columns=["run"]))
    if len(out[1]) != len(out[0]) * 3:
        print("Warning: dim_reduction does not have exactly 3 rows (k = 1, 3, 5) per anisotropy row.")
    return out


def build_isolated_vs_contextualized(paths, exclude=None):
    df = load_simlex_summaries(paths, exclude)
    df["name"] = df["name"].apply(canonicalize_model_name)
    df["standardization"] = standardization_label(df["n_corpus_sentences"])
    df = df.rename(columns={"layer": "layer_num"})
    # Both BaselineLM variants share a model class; the v1 file distinguishes the Pile one.
    df.loc[df["name"] == "BaselineLM - The Pile", "model_cls"] = "BaselineLMModel - the_pile"
    df = add_data_size(df, max_gemstone_step=None).rename(columns={"data_size": "Data Scale"})
    cols = ["name", "model_cls", "embedding_mode", "spearman", "layer_num", "standardization", "Data Scale"]
    return df[cols].sort_values(["name", "layer_num", "embedding_mode", "standardization"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simlex-summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--anisotropy", nargs="+", type=Path, required=True)
    parser.add_argument("--dim-reduction", nargs="+", type=Path, required=True)
    parser.add_argument("--isolated-summaries", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/assembled"))
    parser.add_argument("--exclude", help="Regex; drop runs whose model class (SimLex) or run name (geometry) matches.")
    args = parser.parse_args()

    simlex_df = build_simlex_results(args.simlex_summaries, args.exclude)
    anisotropy_df, dim_reduction_df = build_geometry(args.anisotropy, args.dim_reduction, args.exclude)

    gemstone_steps = simlex_df.loc[simlex_df["name"].str.contains("Gemstone"), "step"]
    max_gemstone_step = gemstone_steps.max() if len(gemstone_steps) else np.nan
    for df in (simlex_df, anisotropy_df, dim_reduction_df):
        add_data_size(df, max_gemstone_step)

    anisotropy_df = anisotropy_df[["name", "layer", "step", "tokens_seen", "data_size", "anisotropy"]]
    dim_reduction_df = dim_reduction_df[["name", "layer", "step", "tokens_seen", "data_size", "k", "r2"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in [
        ("simlex_results.csv", simlex_df),
        ("anisotropy.csv", anisotropy_df),
        ("dim_reduction.csv", dim_reduction_df),
    ]:
        df.sort_values(by=["name", "layer", "step"]).to_csv(args.out_dir / filename, index=False)
        print(f"Wrote {args.out_dir / filename} ({len(df)} rows)")

    if args.isolated_summaries:
        iso_df = build_isolated_vs_contextualized(args.isolated_summaries, args.exclude)
        iso_df.to_csv(args.out_dir / "isolated_vs_contextualized.csv", index=False)
        print(f"Wrote {args.out_dir / 'isolated_vs_contextualized.csv'} ({len(iso_df)} rows)")


if __name__ == "__main__":
    main()

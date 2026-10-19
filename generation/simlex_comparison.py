"""Score SimLex-999 with and without standardization, per layer, for Hugging Face models.

Usage (from the repository root):
    python generation/simlex_comparison.py --config generation/configs/smoke_test.json

Each config lists models (Hugging Face IDs, optionally with a revision for a
training checkpoint). "corpus_sizes": [null, "all"] produces both conditions
reported in the paper: null = Not Standardized, "all" = Standardized, with
mean and standard deviation fit on every token of data/inputs/corpus.txt.
Results go to results/simlex/<timestamp>/summary.csv.

Supports single-process runs and torchrun/SLURM multi-process runs.
"""
import argparse
from pathlib import Path
import json
import csv
import os
from datetime import datetime

import torch
import torch.distributed as dist
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM

from hf_wrapper import HFModelWrapper
from run_simlex import load_simlex_data, load_corpus_sentences, run_simlex_evaluation_all_layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
SIMLEX_PATH = INPUTS_DIR / "SimLex-999.txt"  # download separately; see generation/README.md
CONTEXT_SENTENCES_PATH = INPUTS_DIR / "simlex_context.json"
CORPUS_SENTENCES_PATH = INPUTS_DIR / "corpus.txt"

RESULTS_ROOT = PROJECT_ROOT / "results" / "simlex"


def setup_distributed():
    """Initialize the distributed process group. Supports torchrun and SLURM environments.
    Returns (rank, world_size, local_rank). When not distributed, returns (0, 1, 0).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        hostnames = os.environ.get("SLURM_JOB_NODELIST", "localhost")
        if "[" in hostnames:
            master_addr = hostnames.split("[")[0]
        else:
            master_addr = hostnames.split(",")[0] if "," in hostnames else hostnames
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    else:
        return 0, 1, 0

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, world_size, local_rank


def setup_device(local_rank):
    """Return the appropriate torch device for this process."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON run config.")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = setup_device(local_rank)
    is_main = rank == 0

    if is_main:
        print(f"Running with {world_size} process(es) on device {device}", flush=True)

    # All ranks load the config
    with open(args.config) as f:
        _runs_config = json.load(f)

    CORPUS_SIZES = _runs_config["corpus_sizes"]
    EMBEDDING_MODE = _runs_config.get("embedding_mode", "contextualized")
    if EMBEDDING_MODE not in ("isolated", "contextualized"):
        raise ValueError(f"embedding_mode must be 'isolated' or 'contextualized', got {EMBEDDING_MODE!r}")
    if is_main:
        print(f"Embedding mode: {EMBEDDING_MODE}", flush=True)

    RUNS = []
    for r in _runs_config["runs"]:
        if r.get("type") == "pretrained_hf":
            RUNS.append({
                "type": "pretrained_hf",
                "name": r["name"],
                "model_id": r["model_id"],
                "tokenizer_id": r["tokenizer"],
                "revision": r.get("revision"),
            })
        else:
            raise ValueError(
                f"Run {r.get('name')!r} is not a pretrained Hugging Face model. Locally trained "
                "checkpoints (the paper's BaselineLM) are not part of this release."
            )

    # Only rank 0 creates the output directory
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_ROOT / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {output_dir}", flush=True)

    # All ranks load data
    if is_main:
        print("Loading SimLex-999...", flush=True)
    simlex_data = load_simlex_data(SIMLEX_PATH)

    if is_main:
        print("Loading context sentences...", flush=True)
    with open(CONTEXT_SENTENCES_PATH) as f:
        context_sentences = json.load(f)

    if is_main:
        print("Loading corpus sentences...", flush=True)
    with open(CORPUS_SENTENCES_PATH) as f:
        all_corpus_sentences = [line.rstrip("\n") for line in f if line.strip()]
    if is_main:
        print(f"Total corpus sentences: {len(all_corpus_sentences)}", flush=True)

    # Precompute corpus sentence lists once so all runs use identical sets.
    corpus_sentences_map = {}  # dict[str, list | None]
    for corpus_size in CORPUS_SIZES:
        if corpus_size is None:
            corpus_label = "none"
            corpus_sentences = None
        elif corpus_size == "all":
            corpus_label = "all"
            corpus_sentences = all_corpus_sentences
        else:
            corpus_label = str(corpus_size)
            corpus_sentences = load_corpus_sentences(CORPUS_SENTENCES_PATH, corpus_size)

        # Broadcast from rank 0 so all ranks use the same sentences and produce
        # consistent z-scores.
        if world_size > 1 and corpus_sentences is not None:
            container = [corpus_sentences if is_main else None]
            dist.broadcast_object_list(container, src=0)
            corpus_sentences = container[0]

        corpus_sentences_map[corpus_label] = corpus_sentences

    summary_rows = []

    for run in RUNS:
        try:
            if is_main:
                print(f"\n--- {run['name']} ---", flush=True)

            if is_main:
                print(f"  Loading tokenizer from {run['tokenizer_id']}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(run["tokenizer_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if run["type"] == "pretrained_hf":
                if is_main:
                    revision_str = f" @ {run['revision']}" if run.get("revision") else ""
                    print(f"  Loading pretrained model {run['model_id']}{revision_str}...", flush=True)
                model = HFModelWrapper(AutoModelForCausalLM.from_pretrained(
                    run["model_id"], revision=run.get("revision"), trust_remote_code=True
                ))
            else:
                raise ValueError(f"Unsupported run type: {run['type']!r}")
            model = model.to(device)

            for corpus_size in CORPUS_SIZES:
                if corpus_size is None:
                    corpus_label = "none"
                elif corpus_size == "all":
                    corpus_label = "all"
                else:
                    corpus_label = str(corpus_size)

                corpus_sentences = corpus_sentences_map[corpus_label]

                if is_main:
                    print(f"  corpus_sentences={corpus_label}", flush=True)

                # Each rank scores its own slice of word pairs
                rank_data = simlex_data.iloc[rank::world_size]
                rank_df, _ = run_simlex_evaluation_all_layers(
                    model, tokenizer, rank_data,
                    context_sentences=context_sentences if EMBEDDING_MODE == "contextualized" else None,
                    corpus_sentences=corpus_sentences,
                )

                # Gather all partial DataFrames to rank 0, then recompute Spearman on the full set
                if world_size > 1:
                    all_dfs = [None] * world_size if is_main else None
                    dist.gather_object(rank_df, all_dfs, dst=0)
                    if is_main:
                        results_df = pd.concat(all_dfs, ignore_index=True)
                else:
                    results_df = rank_df

                if is_main:
                    safe_name = run["name"].replace(" ", "_").replace("(", "").replace(")", "")
                    results_path = output_dir / f"{safe_name}_corpus{corpus_label}_results.csv"
                    results_df.to_csv(results_path, index=False)

                    model_cls_name = run["model_id"] if run["type"] == "pretrained_hf" else run["model_cls"].__name__
                    if run["type"] == "pretrained_hf" and run.get("revision"):
                        checkpoint_str = f"{run['model_id']}@{run['revision']}"
                    else:
                        checkpoint_str = str(run.get("checkpoint", run.get("model_id", "")))

                    layer_cols = sorted(
                        [c for c in results_df.columns if c.startswith("score_layer_")],
                        key=lambda c: int(c.split("_")[-1]),
                    )
                    range_cols = sorted(
                        [c for c in results_df.columns if c.startswith("score_layers_0_")],
                        key=lambda c: int(c.split("_")[-1]),
                    )

                    for col in layer_cols:
                        layer_label = col[len("score_"):]  # e.g. "layer_0"
                        spearman_corr = spearmanr(results_df["human_score"], results_df[col]).statistic
                        print(f"  {layer_label}: Spearman={spearman_corr:.4f}", flush=True)
                        summary_rows.append({
                            "name": run["name"],
                            "model_cls": model_cls_name,
                            "checkpoint": checkpoint_str,
                            "layer": layer_label,
                            "n_corpus_sentences": corpus_label,
                            "embedding_mode": EMBEDDING_MODE,
                            "spearman": spearman_corr,
                            "results_path": str(results_path),
                        })

                    for col in range_cols:
                        layer_label = col[len("score_"):]  # e.g. "layers_0_4"
                        spearman_corr = spearmanr(results_df["human_score"], results_df[col]).statistic
                        print(f"  {layer_label}: Spearman={spearman_corr:.4f}", flush=True)
                        summary_rows.append({
                            "name": run["name"],
                            "model_cls": model_cls_name,
                            "checkpoint": checkpoint_str,
                            "layer": layer_label,
                            "n_corpus_sentences": corpus_label,
                            "embedding_mode": EMBEDDING_MODE,
                            "spearman": spearman_corr,
                            "results_path": str(results_path),
                        })
        except Exception as e:
            if is_main:
                print(f"Error processing run {run['name']}: {e}", flush=True)

    if is_main:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "model_cls", "checkpoint", "layer", "n_corpus_sentences", "embedding_mode", "spearman", "results_path"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved to {summary_path}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

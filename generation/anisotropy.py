"""
Given a model and a tokenizer, calculates the anisotropy and
the per-dimension cosine similarity contribution.

We use 500k randomly sampled token pairs (seed 42) from our Wikipedia subset,
data/inputs/corpus.txt. Also computes rogue dimensionality: R^2 between full-space
cosine similarity and cosine similarity with the top k contributing dimensions
removed, for k = 1, 3, 5.

Usage (from the repository root):
    python generation/anisotropy.py --config generation/configs/smoke_test.json

Results are appended to <output_dir>/anisotropy.csv, dim_reduction_r2.csv, and
per_dim_contribution.csv (output_dir defaults to results/anisotropy).
"""

import argparse

import csv
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from hf_wrapper import HFModelWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runs_from_config(config_path):
    """Parse anisotropy_config.json and return a list of resolved run dicts.

    Each returned dict has at minimum:
        "name"        - display name for the run
        "type"        - "checkpoint" or "pretrained_hf"
        "tokenizer_id"- path or HF identifier passed to AutoTokenizer

    Pretrained-HF runs additionally have:
        "model_id"    - HF model identifier
    """
    with open(config_path) as f:
        cfg = json.load(f)

    runs = []
    for run in cfg["runs"]:
        if run.get("type") == "pretrained_hf":
            revision = run.get("revision")
            name = f"{run['name']}@{revision}" if revision else run["name"]
            runs.append({
                "type": "pretrained_hf",
                "name": name,
                "model_id": run["model_id"],
                "tokenizer_id": run["tokenizer"],
                "revision": revision,
            })
        else:
            raise ValueError(
                f"Run {run.get('name')!r} is not a pretrained Hugging Face model. Locally trained "
                "checkpoints (the paper's BaselineLM) are not part of this release."
            )
    return cfg, runs


def load_model_and_tokenizer(run, device):
    """Load and return (model, tokenizer) for a resolved run dict."""
    tokenizer = AutoTokenizer.from_pretrained(run["tokenizer_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if run["type"] == "pretrained_hf":
        model = HFModelWrapper(AutoModelForCausalLM.from_pretrained(
            run["model_id"], revision=run.get("revision"), trust_remote_code=True
        ))
    else:
        raise ValueError(f"Unsupported run type: {run['type']!r}")

    model = model.to(device)
    return model, tokenizer


def build_pairs(model, tokenizer, n_pairs, corpus_path):
    """Tokenize the corpus and sample random token pairs.

    Args:
        model: Used only to read config.max_seq_length for truncation (if present).
        tokenizer: The tokenizer.
        n_pairs: Number of random token pairs to sample.
        corpus_path: Path to the corpus text file (one sentence per line).

    Returns:
        Tuple of (tokenized_corpus, pairs) where tokenized_corpus is a list of token ID
        tensors and pairs is a list of (sent1_id, token1_id, sent2_id, token2_id) tuples.
    """
    max_length = getattr(getattr(model, "config", None), "max_seq_length", None)
    tokenized_corpus = tokenize_corpus(corpus_path, tokenizer)
    if max_length is not None:
        tokenized_corpus = [t[:max_length] for t in tokenized_corpus]
    pairs = sample_random_token_pairs(tokenized_corpus, n_pairs, seed=42)
    return tokenized_corpus, pairs


def build_hidden_states_cache(model, tokenizer, tokenized_corpus, pairs, device):
    """Run the model on all sentences referenced by pairs and cache their hidden states.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer (used for pad_token_id).
        tokenized_corpus: List of token ID tensors (as returned by build_pairs).
        pairs: List of (sent1_id, token1_id, sent2_id, token2_id) tuples.
        device: torch.device.

    Returns:
        Dict mapping sent_id -> {layer_idx: tensor(seq_len, d_model)} stored on CPU.
    """
    unique_sentences = set()
    for sent1_id, _, sent2_id, _ in pairs:
        unique_sentences.add(sent1_id)
        unique_sentences.add(sent2_id)

    hidden_states_cache = {}  # sent_id -> {layer_idx: tensor(T, d_model)} on CPU
    batch_size = 32
    unique_list = sorted(unique_sentences)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    was_training = model.training
    model.eval()
    print(f"  Computing hidden states for {len(unique_list)} unique sentences...", flush=True)
    with torch.no_grad():
        for i in range(0, len(unique_list), batch_size):
            if i % 2000 == 0 and i > 0:
                print(f"  {i}/{len(unique_list)} sentences processed", flush=True)
            batch_ids = unique_list[i:i + batch_size]
            batch_tokens = [tokenized_corpus[sid] for sid in batch_ids]
            lengths = [t.size(0) for t in batch_tokens]
            max_len = max(lengths)

            input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
            for j, (tokens, length) in enumerate(zip(batch_tokens, lengths)):
                input_ids[j, :length] = tokens
                attention_mask[j, :length] = 1

            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

            for j, sid in enumerate(batch_ids):
                hidden_states_cache[sid] = {
                    layer_idx: hidden_states[j, :lengths[j]].cpu()
                    for layer_idx, hidden_states in enumerate(outputs.all_hidden_states)
                }
    if was_training:
        model.train()

    return hidden_states_cache

def compute_anisotropy_from_cache(hidden_states_cache, pairs):
    """Compute anisotropy and per-dimension cosine similarity contribution from a prebuilt cache.

    For each pair (u, v), the cosine similarity decomposes as:
        cos(u, v) = sum_d (u_norm_d * v_norm_d)
    so the mean cosine similarity (anisotropy) = sum_d mean(u_norm_d * v_norm_d),
    and the per-dimension contribution for dim d = mean(u_norm_d * v_norm_d).

    Pairs are processed in chunks to bound peak memory usage.

    Args:
        hidden_states_cache: Dict mapping sent_id -> {layer_idx: tensor(seq_len, d_model)} on CPU,
                             as returned by build_hidden_states_cache.
        pairs: List of (sent1_id, token1_id, sent2_id, token2_id) tuples.

    Returns:
        list of dicts with keys "anisotropy" and "per_dim_contribution". One dict per layer.
    """
    n_layers = len(next(iter(hidden_states_cache.values())))
    d_model = next(iter(hidden_states_cache.values()))[0].size(-1)

    cos_sums = [0.0] * n_layers
    dim_sums = [torch.zeros(d_model) for _ in range(n_layers)]
    n_total = len(pairs)
    chunk_size = 10_000

    for chunk_start in range(0, n_total, chunk_size):
        chunk = pairs[chunk_start:chunk_start + chunk_size]

        # Collect (u, v) tensors for all layers in a single pass over the chunk.
        u_s: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        v_s: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        for sent1_id, tok1_id, sent2_id, tok2_id in chunk:
            cache1 = hidden_states_cache[sent1_id]
            cache2 = hidden_states_cache[sent2_id]
            for layer_idx in range(n_layers):
                u_s[layer_idx].append(cache1[layer_idx][tok1_id])
                v_s[layer_idx].append(cache2[layer_idx][tok2_id])

        for layer_idx in range(n_layers):
            u = torch.stack(u_s[layer_idx])  # (chunk_len, d)
            v = torch.stack(v_s[layer_idx])
            u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-8)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)
            per_dim = u * v  # (chunk_len, d): per-dim contribution per pair
            cos_sums[layer_idx] += per_dim.sum(dim=1).sum().item()
            dim_sums[layer_idx] += per_dim.sum(dim=0)

    return [
        {
            "anisotropy": cos_sums[layer_idx] / n_total,
            "per_dim_contribution": (dim_sums[layer_idx] / n_total).tolist(),
        }
        for layer_idx in range(n_layers)
    ]

def sample_random_token_pairs(tokenized_corpus, n_pairs, seed=None):
    """Sample n_pairs random token pairs from the tokenized corpus.

    Args:
        tokenized_corpus: List of token ID tensors.
        n_pairs: Number of random pairs to sample.
        seed: Optional random seed for reproducibility.
    Returns:
        List of (sent1_id, token1_id, sent2_id, token2_id) tuples.
    """
    if seed is not None:
        random.seed(seed)

    pairs = set()
    while len(pairs) < n_pairs:
        sent1_id = random.randint(0, len(tokenized_corpus) - 1)
        sent2_id = random.randint(0, len(tokenized_corpus) - 1)
        token1_id = random.randint(0, len(tokenized_corpus[sent1_id]) - 1)
        token2_id = random.randint(0, len(tokenized_corpus[sent2_id]) - 1)
        pairs.add((sent1_id, token1_id, sent2_id, token2_id))
    return list(pairs)

def tokenize_corpus(corpus_path, tokenizer):
    """
    Tokenize the sentences in the corpus and return a list of token ID tensors.

    Args:
        corpus_path: Path to the corpus text file (one sentence per line).
        n_contexts: Number of random contexts to sample.
        tokenizer: The tokenizer to convert sentences to token IDs.
        seed: Optional random seed for reproducibility.
    Returns:
        List of token_ids_tensors.
    """

    # Read the corpus
    with open(corpus_path, "r") as f:
        lines = f.readlines()
    
    token_id_tensors = []
    for line in lines:
        token_ids = tokenizer.encode(line.strip(), add_special_tokens=False)
        token_id_tensors.append(torch.tensor(token_ids, dtype=torch.long))
    return token_id_tensors

def correlate_full_vs_reduced(
    pairs,
    hidden_states_cache,
    sorted_dims: list[list[int]],
    ks: list[int] = [1, 3, 5],
) -> list[dict]:
    """Compute R² between full-space and top-k-excluded cosine similarities, per layer per k.

    For each layer and each k, every pair's cosine similarity is computed twice:
      - full: using all dimensions
      - reduced: using all dimensions *except* the top-k contributors (as ranked by sorted_dims)

    The Pearson r between the two distributions is squared to give R².

    Args:
        pairs: List of (sent1_id, token1_id, sent2_id, token2_id) tuples.
        hidden_states_cache: Dict mapping sent_id -> {layer_idx: tensor(seq_len, d_model)} on CPU,
                             as returned by build_hidden_states_cache.
        sorted_dims: Per-layer list of dimension indices sorted from highest to lowest contribution,
                     as returned by rank_dimensions_by_contribution.
        ks: List of k values specifying how many top dimensions to exclude.

    Returns:
        List of dicts (one per layer), each mapping k -> R² value.
        E.g. [{"k=1": 0.98, "k=3": 0.95, "k=5": 0.91}, ...]
    """
    # sorted_dims has one entry per layer (see rank_dimensions_by_contribution).
    n_layers = len(sorted_dims)
    n_pairs = len(pairs)
    chunk_size = 10_000

    # Accumulate per-layer full and per-k reduced cosine similarities across all pairs.
    full_cos: list[list[float]] = [[] for _ in range(n_layers)]
    reduced_cos: dict[int, list[list[float]]] = {k: [[] for _ in range(n_layers)] for k in ks}

    for chunk_start in range(0, n_pairs, chunk_size):
        chunk = pairs[chunk_start:chunk_start + chunk_size]

        u_s: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        v_s: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        for sent1_id, tok1_id, sent2_id, tok2_id in chunk:
            cache1 = hidden_states_cache[sent1_id]
            cache2 = hidden_states_cache[sent2_id]
            for layer_idx in range(n_layers):
                u_s[layer_idx].append(cache1[layer_idx][tok1_id])
                v_s[layer_idx].append(cache2[layer_idx][tok2_id])

        for layer_idx in range(n_layers):
            u = torch.stack(u_s[layer_idx]).float()  # (chunk_len, d)
            v = torch.stack(v_s[layer_idx]).float()
            u_norm = u / u.norm(dim=1, keepdim=True).clamp(min=1e-8)
            v_norm = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)

            full_cos[layer_idx].extend((u_norm * v_norm).sum(dim=1).tolist())

            for k in ks:
                excluded = set(sorted_dims[layer_idx][:k])
                keep = [d for d in range(u.size(1)) if d not in excluded]
                u_red = u[:, keep]
                v_red = v[:, keep]
                u_red = u_red / u_red.norm(dim=1, keepdim=True).clamp(min=1e-8)
                v_red = v_red / v_red.norm(dim=1, keepdim=True).clamp(min=1e-8)
                reduced_cos[k][layer_idx].extend((u_red * v_red).sum(dim=1).tolist())

    def pearson_r2(xs: list[float], ys: list[float]) -> float:
        x = torch.tensor(xs)
        y = torch.tensor(ys)
        x = x - x.mean()
        y = y - y.mean()
        r = (x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-8)
        return r.item() ** 2

    return [
        {f"k={k}": pearson_r2(full_cos[layer_idx], reduced_cos[k][layer_idx]) for k in ks}
        for layer_idx in range(n_layers)
    ]


def rank_dimensions_by_contribution(results: list[dict]) -> list[list[int]]:
    """Return dimension indices sorted by per-dimension contribution, highest first, per layer.

    Args:
        results: List of per-layer dicts as returned by compute_anisotropy, each containing
                 a "per_dim_contribution" key with a list of floats.

    Returns:
        A list (one entry per layer) of dimension index lists sorted from highest to lowest
        per-dimension contribution.
    """
    result = []
    for layer in results:
        sorted_dims = sorted(range(len(layer["per_dim_contribution"])),
                              key=lambda dim: layer["per_dim_contribution"][dim],
                              reverse=True)
        result.append(sorted_dims)
    return result



def save_dim_reduction_r2(output_dir: Path, run_name: str, r2_results: list[dict]) -> None:
    """Append R² results from correlate_full_vs_reduced to a CSV in output_dir.

    Writes/appends:
      dim_reduction_r2.csv - columns: run, layer, k, r2
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dim_reduction_r2.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run", "layer", "k", "r2"])
        for layer_idx, layer_result in enumerate(r2_results):
            for k_label, r2 in layer_result.items():
                writer.writerow([run_name, layer_idx, k_label, r2])


def save_results(output_dir: Path, run_name: str, results: list[dict]) -> None:
    """Append anisotropy results for one run to CSVs in output_dir.

    Writes/appends two files:
      anisotropy.csv         - columns: run, layer, anisotropy
      per_dim_contribution.csv - columns: run, layer, dim, contribution
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    anisotropy_path = output_dir / "anisotropy.csv"
    per_dim_path = output_dir / "per_dim_contribution.csv"

    write_header_anisotropy = not anisotropy_path.exists()
    write_header_per_dim = not per_dim_path.exists()

    with open(anisotropy_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header_anisotropy:
            writer.writerow(["run", "layer", "anisotropy"])
        for layer_idx, layer_result in enumerate(results):
            writer.writerow([run_name, layer_idx, layer_result["anisotropy"]])

    with open(per_dim_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header_per_dim:
            writer.writerow(["run", "layer", "dim", "contribution"])
        for layer_idx, layer_result in enumerate(results):
            for dim_idx, contrib in enumerate(layer_result["per_dim_contribution"]):
                writer.writerow([run_name, layer_idx, dim_idx, contrib])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON run config.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg, runs = load_runs_from_config(args.config)

    n_pairs = cfg.get("n_pairs", 500_000)
    corpus_path = PROJECT_ROOT / cfg.get("corpus_path", "data/inputs/corpus.txt")
    output_dir = PROJECT_ROOT / cfg.get("output_dir", "results/anisotropy")

    print(f"n_pairs: {n_pairs}", flush=True)
    print(f"corpus_path: {corpus_path}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)

    for run in runs:
        print(f"\n--- {run['name']} ---", flush=True)
        print(f"  Loading tokenizer from {run['tokenizer_id']}...", flush=True)
        model, tokenizer = load_model_and_tokenizer(run, device)
        tokenized_corpus, pairs = build_pairs(model, tokenizer, n_pairs, corpus_path)
        hidden_states_cache = build_hidden_states_cache(model, tokenizer, tokenized_corpus, pairs, device)
        results = compute_anisotropy_from_cache(hidden_states_cache, pairs)
        for layer_idx, layer_result in enumerate(results):
            print(f"  Layer {layer_idx} anisotropy: {layer_result['anisotropy']:.4f}", flush=True)
        save_results(output_dir, run["name"], results)
        print(f"  Results saved to {output_dir}", flush=True)

        print("  Correlating full-space and top-k-excluded cosine similarities (k=1, 3, 5)...", flush=True)
        sorted_dims = rank_dimensions_by_contribution(results)
        r2_results = correlate_full_vs_reduced(pairs, hidden_states_cache, sorted_dims)
        for layer_idx, layer_result in enumerate(r2_results):
            for k_label, r2 in layer_result.items():
                print(f"  Layer {layer_idx} {k_label} R²: {r2:.4f}", flush=True)
        save_dim_reduction_r2(output_dir, run["name"], r2_results)
        print(f"  R² results saved to {output_dir}", flush=True)

if __name__ == "__main__":
    main()

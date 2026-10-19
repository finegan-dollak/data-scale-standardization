import random
import sys
import warnings

import similarity_scoring
import pandas as pd
import torch
from tqdm import tqdm

REQUIRED_COLUMNS = {"word1", "word2", "SimLex999"}


def load_corpus_sentences(sentences_path, n, seed=None):
    """Sample N sentences from a plain-text file (one sentence per line).

    Args:
        sentences_path: Path to a text file with one sentence per line.
        n: Number of sentences to return.
        seed: Optional random seed for reproducibility.

    Returns:
        List of ``n`` sentence strings sampled from the file.

    Raises:
        ValueError: If the file contains fewer lines than ``n``.
    """
    with open(sentences_path) as f:
        candidates = [line.rstrip("\n") for line in f if line.strip()]
    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} sentences in {sentences_path}, "
            f"but {n} were requested."
        )
    rng = random.Random(seed)
    return rng.sample(candidates, n)


def load_simlex_data(simlex_path):
    """Load SimLex-999 data from a TSV file.

    Args:
        simlex_path: Path to the SimLex-999 TSV file.

    Returns:
        DataFrame with one row per word pair and columns including
        'word1', 'word2', and 'SimLex999' (the human similarity score).

    Raises:
        ValueError: If the file is missing any of the expected columns.
    """
    df = pd.read_csv(simlex_path, sep="\t")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"SimLex file is missing expected columns: {missing}")
    return df


def _make_scorer(model, tokenizer, layer, context_sentences, corpus_sentences=None):
    """Build a CosineSimilarityScorer backed by the appropriate embedder.

    If ``corpus_sentences`` is provided, calls ``fit_standardization`` on the
    embedder before returning so that all subsequent embeddings are z-scored.
    """
    if context_sentences is not None:
        embedder = similarity_scoring.ContextualWordEmbedder(model, tokenizer, context_sentences)
    else:
        embedder = similarity_scoring.IsolatedWordEmbedder(model, tokenizer)
    if corpus_sentences is not None:
        embedder.fit_standardization(corpus_sentences)
    return similarity_scoring.CosineSimilarityScorer(embedder, layer=layer)


def run_simlex_evaluation(model, tokenizer, simlex_data, layer=-1, context_sentences=None, corpus_sentences=None):
    """Evaluate a model on SimLex-999 word pairs at a single layer.

    Args:
        model: A model compatible with CosineSimilarityScorer.
        tokenizer: Tokenizer compatible with the model.
        simlex_data: DataFrame as returned by load_simlex_data, with columns
            'word1', 'word2', and 'SimLex999'.
        layer: Which layer's hidden states to use for scoring (default -1 for last layer).
        context_sentences: Optional dict mapping words to lists of context sentences.
            If provided, ContextualWordEmbedder is used; otherwise IsolatedWordEmbedder.
        corpus_sentences: Optional list of sentences used to fit per-layer z-score
            standardization before scoring. See ``WordEmbedder.fit_standardization``.

    Returns:
        Tuple of (results_df, spearman_corr) where results_df is a DataFrame with
        columns 'word1', 'word2', 'human_score', 'model_score', and 'layer', and
        spearman_corr is the Spearman correlation between human and model scores.

    Raises:
        ValueError: If all word pairs were skipped and no results could be computed.
    """
    scorer = _make_scorer(model, tokenizer, layer, context_sentences, corpus_sentences)
    results = []
    n_skipped = 0
    for _, row in tqdm(simlex_data.iterrows(), total=len(simlex_data), desc="Scoring pairs", disable=not sys.stderr.isatty()):
        word1, word2, human_score = row["word1"], row["word2"], row["SimLex999"]
        try:
            model_score = scorer.score(word1, word2)
            if not torch.isfinite(torch.tensor(model_score)):
                raise ValueError(f"Non-finite score for ('{word1}', '{word2}')")
        except ValueError as e:
            warnings.warn(str(e))
            n_skipped += 1
            continue
        results.append({
            "word1": word1,
            "word2": word2,
            "human_score": human_score,
            "model_score": model_score,
            "layer": layer
        })

    if n_skipped > 0:
        warnings.warn(f"Skipped {n_skipped} of {len(simlex_data)} pairs due to scoring errors.")

    if len(results) == 0:
        raise ValueError("All word pairs were skipped; cannot compute Spearman correlation.")

    df = pd.DataFrame(results)
    spearman_corr = df[["human_score", "model_score"]].corr(method="spearman").iloc[0, 1]
    return df, spearman_corr


def run_simlex_evaluation_all_layers(model, tokenizer, simlex_data, context_sentences=None, corpus_sentences=None):
    """Evaluate a model on SimLex-999 across all individual layers and all
    cumulative layer ranges in a single forward pass per word.

    Individual layers: layer 0, 1, ..., N-1.
    Cumulative ranges: mean(layers 0..n) for n in range(2, N+1), i.e.
    ``all_hidden_states[0:n]`` averaged, written as ``layers_0_{n}``.

    Args:
        model: A model compatible with CosineSimilarityScorer.
        tokenizer: Tokenizer compatible with the model.
        simlex_data: DataFrame as returned by load_simlex_data.
        context_sentences: Optional dict mapping words to lists of context sentences.
        corpus_sentences: Optional list of sentences used to fit per-layer z-score
            standardization before scoring.

    Returns:
        Tuple of (results_df, spearman_by_spec) where:
        - results_df has columns 'word1', 'word2', 'human_score',
          'score_layer_{i}' for each layer i, and
          'score_layers_0_{n}' for each cumulative range (n >= 2).
        - spearman_by_spec maps spec label (e.g. ``'layer_0'``,
          ``'layers_0_3'``) to Spearman correlation.

    Raises:
        ValueError: If all word pairs were skipped.
    """
    import torch.nn.functional as F

    if context_sentences is not None:
        embedder = similarity_scoring.ContextualWordEmbedder(model, tokenizer, context_sentences)
    else:
        embedder = similarity_scoring.IsolatedWordEmbedder(model, tokenizer)
    if corpus_sentences is not None:
        embedder.fit_standardization(corpus_sentences)

    results = []
    n_skipped = 0
    n_layers = None

    for _, row in tqdm(simlex_data.iterrows(), total=len(simlex_data), desc="Scoring pairs", disable=not sys.stderr.isatty()):
        word1, word2, human_score = row["word1"], row["word2"], row["SimLex999"]
        try:
            embs1 = embedder.embed_all_layers(word1)
            embs2 = embedder.embed_all_layers(word2)

            if n_layers is None:
                n_layers = len(embs1)

            record = {"word1": word1, "word2": word2, "human_score": human_score}

            # Individual layers
            for i in sorted(embs1):
                v1, v2 = embs1[i].cpu(), embs2[i].cpu()
                sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                if not torch.isfinite(torch.tensor(sim)):
                    raise ValueError(f"Non-finite score for ('{word1}', '{word2}') at layer {i}")
                record[f"score_layer_{i}"] = sim

            # Cumulative ranges: mean of layers 0..n for n in 2..n_layers
            for n in range(2, n_layers + 1):
                v1 = torch.stack([embs1[i].cpu() for i in range(n)]).mean(dim=0)
                v2 = torch.stack([embs2[i].cpu() for i in range(n)]).mean(dim=0)
                sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                if not torch.isfinite(torch.tensor(sim)):
                    raise ValueError(f"Non-finite score for ('{word1}', '{word2}') at layers 0..{n}")
                record[f"score_layers_0_{n}"] = sim

            results.append(record)
        except ValueError as e:
            warnings.warn(str(e))
            n_skipped += 1
            continue

    if n_skipped > 0:
        warnings.warn(f"Skipped {n_skipped} of {len(simlex_data)} pairs due to scoring errors.")

    if not results:
        raise ValueError("All word pairs were skipped; cannot compute Spearman correlation.")

    df = pd.DataFrame(results)

    spearman_by_spec = {}
    if n_layers is not None:
        for i in range(n_layers):
            col = f"score_layer_{i}"
            spearman_by_spec[f"layer_{i}"] = df[["human_score", col]].corr(method="spearman").iloc[0, 1]
        for n in range(2, n_layers + 1):
            col = f"score_layers_0_{n}"
            spearman_by_spec[f"layers_0_{n}"] = df[["human_score", col]].corr(method="spearman").iloc[0, 1]

    return df, spearman_by_spec


def run_simlex_evaluation_layers(model, tokenizer, simlex_data, layers=None, context_sentences=None, corpus_sentences=None):
    """Evaluate a model on SimLex-999 across multiple layers in a single pass.

    Each word pair triggers at most two model forward passes (one per word),
    and all requested layers are scored from those cached embeddings.

    Args:
        model: A model compatible with CosineSimilarityScorer.
        tokenizer: Tokenizer compatible with the model.
        simlex_data: DataFrame as returned by load_simlex_data.
        layers: List of layer indices to evaluate. Defaults to all layers.
        context_sentences: Optional dict mapping words to lists of context sentences.
        corpus_sentences: Optional list of sentences used to fit per-layer z-score
            standardization before scoring. See ``WordEmbedder.fit_standardization``.

    Returns:
        Tuple of (results_df, spearman_by_layer) where results_df has columns
        'word1', 'word2', 'human_score', and one 'score_layer_{i}' column per
        layer, and spearman_by_layer is a dict mapping layer index to Spearman
        correlation.

    Raises:
        ValueError: If all word pairs were skipped.
    """
    # Use layer=-1 as a placeholder; score_layers ignores self.layer
    scorer = _make_scorer(model, tokenizer, -1, context_sentences, corpus_sentences)
    results = []
    n_skipped = 0

    for _, row in tqdm(simlex_data.iterrows(), total=len(simlex_data), desc="Scoring pairs", disable=not sys.stderr.isatty()):
        word1, word2, human_score = row["word1"], row["word2"], row["SimLex999"]
        try:
            layer_scores = scorer.score_layers(word1, word2, layers=layers)
            for score in layer_scores.values():
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError(f"Non-finite score for ('{word1}', '{word2}')")
        except ValueError as e:
            warnings.warn(str(e))
            n_skipped += 1
            continue

        record = {"word1": word1, "word2": word2, "human_score": human_score}
        for layer_idx, score in layer_scores.items():
            record[f"score_layer_{layer_idx}"] = score
        results.append(record)

    if n_skipped > 0:
        warnings.warn(f"Skipped {n_skipped} of {len(simlex_data)} pairs due to scoring errors.")

    if not results:
        raise ValueError("All word pairs were skipped; cannot compute Spearman correlation.")

    df = pd.DataFrame(results)
    scored_layers = [int(col.split("_")[-1]) for col in df.columns if col.startswith("score_layer_")]
    spearman_by_layer = {
        layer_idx: df[["human_score", f"score_layer_{layer_idx}"]].corr(method="spearman").iloc[0, 1]
        for layer_idx in scored_layers
    }
    return df, spearman_by_layer

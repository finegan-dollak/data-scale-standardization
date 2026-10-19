# Similarity scoring
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import warnings
import torch


class WordEmbedder(ABC):
    """Abstract base class that runs a model forward pass and returns per-layer word embeddings.

    Subclasses implement ``_compute_all_layers``, which tokenizes a word (with or
    without context), runs the model once, and returns a mean-pooled embedding
    for every layer.

    Results are cached by word so the model is run at most once per unique word.
    """

    def __init__(self):
        self._cache: dict[str, dict[int, torch.Tensor]] = {}
        self._std_params: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None

    def embed_all_layers(self, word: str) -> dict[int, torch.Tensor]:
        """Return mean-pooled embeddings for all layers for a word.

        If standardization has been fitted via ``fit_standardization``, each
        embedding is z-scored using the per-layer (μ, σ) computed from the
        corpus.

        Args:
            word: Word to embed.

        Returns:
            Dict mapping layer index (0 = embedding layer, 1..n = transformer
            block outputs) to a 1-D tensor of shape ``(d_model,)``.
        """
        if word not in self._cache:
            self._cache[word] = self._compute_all_layers(word)
        raw = self._cache[word]
        if self._std_params is None:
            return raw
        return {
            layer_idx: (vec - self._std_params[layer_idx][0]) / self._std_params[layer_idx][1]
            for layer_idx, vec in raw.items()
        }

    @abstractmethod
    def _compute_all_layers(self, word: str) -> dict[int, torch.Tensor]:
        """Run the model and return per-layer embeddings. Called at most once per word."""
        pass

    @abstractmethod
    def _run_sentence(self, sentence: str) -> dict[int, torch.Tensor]:
        """Tokenize ``sentence``, run one forward pass, and return all token hidden
        states per layer.

        Returns:
            Dict mapping layer index to a 2-D tensor of shape ``(n_tokens, d_model)``.
        """
        pass

    def fit_standardization(self, corpus_sentences: list[str]) -> None:
        """Compute per-layer (μ, σ) from a corpus of sentences and enable z-scoring.

        After calling this, ``embed_all_layers`` returns z-scored embeddings:
        ``z = (x − μ) / σ``, where μ and σ are computed per dimension across
        all tokens in all ``corpus_sentences``.  A separate (μ_i, σ_i) pair is
        stored for each layer *i*.

        Uses population standard deviation (``unbiased=False``).  Dimensions
        with σ < 1e-8 are clamped to 1e-8 to avoid division by zero.

        Args:
            corpus_sentences: Sentences to use as the standardization corpus.
                All tokens in each sentence contribute to (μ, σ).
        """
        layer_vecs: dict[int, list[torch.Tensor]] = {}
        for i, sentence in enumerate(corpus_sentences):
            if i % 1000 == 0:
                print(f"  Standardization - sentences processed: {i}/{len(corpus_sentences)}")
            for layer_idx, token_embs in self._run_sentence(sentence).items():
                layer_vecs.setdefault(layer_idx, []).append(token_embs)

        self._std_params = {}
        for layer_idx, vecs in layer_vecs.items():
            stacked = torch.cat(vecs, dim=0)     # (total_tokens, d)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0, unbiased=False).clamp(min=1e-8)
            self._std_params[layer_idx] = (mean, std)

    def clear_standardization(self) -> None:
        """Remove fitted standardization parameters; embeddings revert to raw."""
        self._std_params = None

    def clear_cache(self):
        """Discard all cached embeddings and fitted standardization parameters."""
        self._cache.clear()
        self._std_params = None


class IsolatedWordEmbedder(WordEmbedder):
    """Embeds a word in isolation (no context sentences).

    Tokenizes the word without special tokens, runs a single forward pass, and
    mean-pools each layer's hidden states over the token dimension.

    Args:
        model: Model whose forward pass returns output with ``all_hidden_states``.
        tokenizer: Tokenizer compatible with the model.
    """

    def __init__(self, model, tokenizer, batch_size: int = 32):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = getattr(getattr(model, "config", None), "max_seq_length", None)

    def fit_standardization(self, corpus_sentences: list[str]) -> None:
        device = next(self.model.parameters()).device
        layer_vecs: dict[int, list[torch.Tensor]] = {}

        was_training = self.model.training
        self.model.eval()

        for i in range(0, len(corpus_sentences), self.batch_size):
            if i % 1000 == 0:
                print(f"  Standardization - sentences processed: {i}/{len(corpus_sentences)}", flush=True)
            batch = corpus_sentences[i:i + self.batch_size]
            tokenized = self.tokenizer(
                batch,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            attention_mask = tokenized["attention_mask"]  # (B, T)
            model_input_keys = {"input_ids", "attention_mask"}
            inputs = {k: v.to(device) for k, v in tokenized.items() if k in model_input_keys}
            with torch.no_grad():
                outputs = self.model(**inputs)
            for layer_idx, hs in enumerate(outputs.all_hidden_states):
                # hs: (B, T, d); collect only non-padding tokens per sentence
                for b in range(hs.size(0)):
                    mask = attention_mask[b].bool()
                    layer_vecs.setdefault(layer_idx, []).append(hs[b][mask])
                    # Could put this on CPU if we increase the size of corpus substantially,
                    # but it may be more efficient to keep it on GPU until we concatenate and compute mean/std

        if was_training:
            self.model.train()

        self._std_params = {}
        for layer_idx, vecs in layer_vecs.items():
            stacked = torch.cat(vecs, dim=0)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0, unbiased=False).clamp(min=1e-8)
            self._std_params[layer_idx] = (mean, std)
        print("  Standardization fitting complete.", flush=True)

    def _compute_all_layers(self, word: str) -> dict[int, torch.Tensor]:
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer(word, return_tensors="pt", add_special_tokens=False)
        if len(tokenized["input_ids"][0]) == 0:
            raise ValueError(f"Word '{word}' produced no tokens after tokenization.")
        model_input_keys = {"input_ids", "attention_mask"}
        inputs = {k: v.to(device) for k, v in tokenized.items() if k in model_input_keys}

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        if was_training:
            self.model.train()

        # Mean-pool over tokens for each layer; squeeze the batch dimension
        return {
            i: hs.squeeze(0).mean(dim=0)
            for i, hs in enumerate(outputs.all_hidden_states)
        }

    def _run_sentence(self, sentence: str) -> dict[int, torch.Tensor]:
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True,
                                   truncation=True, max_length=self.max_length)
        model_input_keys = {"input_ids", "attention_mask"}
        inputs = {k: v.to(device) for k, v in tokenized.items() if k in model_input_keys}

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        if was_training:
            self.model.train()

        # Return all token hidden states per layer; squeeze the batch dimension
        return {i: hs.squeeze(0) for i, hs in enumerate(outputs.all_hidden_states)}


class ContextualWordEmbedder(WordEmbedder):
    """Embeds a word using its context sentences.

    For each word, runs the model on each context sentence, extracts the hidden
    states for the tokens that correspond to the word (via character-level offset
    mapping), and mean-pools first over word tokens then over sentences. Returns
    one embedding vector per layer.

    Falls back to isolated embedding if no usable context sentences exist.

    Requires a tokenizer that supports ``return_offsets_mapping=True`` (all
    HuggingFace fast tokenizers do).

    Args:
        model: Model whose forward pass returns output with ``all_hidden_states``.
        tokenizer: Fast tokenizer compatible with the model.
        context_sentences: Dict mapping each word to a list of context sentences.
        batch_size: Number of sentences per forward pass.
        max_length: Max token length for truncation. If None, inferred from
            ``model.config.max_seq_length`` if available, otherwise no truncation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        context_sentences: dict[str, list[str]],
        batch_size: int = 10,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.context_sentences = context_sentences
        self.batch_size = batch_size
        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = getattr(getattr(model, "config", None), "max_seq_length", None)
        self._isolated = IsolatedWordEmbedder(model, tokenizer, batch_size=batch_size)

    def _run_sentence(self, sentence: str) -> dict[int, torch.Tensor]:
        return self._isolated._run_sentence(sentence)

    def fit_standardization(self, corpus_sentences: list[str]) -> None:
        self._isolated.fit_standardization(corpus_sentences)
        self._std_params = self._isolated._std_params

    def _compute_all_layers(self, word: str) -> dict[int, torch.Tensor]:
        sentences = self.context_sentences.get(word, [])
        if not sentences:
            warnings.warn(f"No context sentences for '{word}'; falling back to isolated embedding.")
            return self._isolated._compute_all_layers(word)

        # layer_idx -> list of (n_valid, d_model) tensors, one per batch
        layer_batches: dict[int, list[torch.Tensor]] = {}
        for i in range(0, len(sentences), self.batch_size):
            batch_result = self._embed_batch(word, sentences[i:i + self.batch_size])
            for layer_idx, embs in batch_result.items():
                layer_batches.setdefault(layer_idx, []).append(embs)

        if not layer_batches:
            warnings.warn(
                f"No usable context sentences remain for '{word}' after skipping bad sentences; "
                "falling back to isolated embedding."
            )
            return self._isolated._compute_all_layers(word)

        # Mean over all valid sentences for each layer -> (d_model,)
        return {
            layer_idx: torch.cat(embs_list, dim=0).mean(dim=0)
            for layer_idx, embs_list in layer_batches.items()
        }

    def _embed_batch(self, word: str, sentences: list[str]) -> dict[int, torch.Tensor]:
        """Run one forward pass over a batch of sentences.

        Returns:
            Dict mapping layer index to a tensor of shape ``(n_valid, d_model)``
            where ``n_valid`` is the number of sentences that yielded at least one
            token for the word. Returns an empty dict if no sentences were usable.
        """
        device = next(self.model.parameters()).device
        word_lower = word.lower()

        valid_sentences = []
        all_word_spans = []
        for sentence in sentences:
            spans = []
            start = 0
            while True:
                idx = sentence.lower().find(word_lower, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(word)))
                start = idx + 1
            if not spans:
                warnings.warn(f"Word '{word}' not found in sentence: {sentence!r}; skipping.")
                continue
            valid_sentences.append(sentence)
            all_word_spans.append(spans)

        if not valid_sentences:
            return {}

        tokenized = self.tokenizer(
            valid_sentences,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        offset_mapping = tokenized.pop("offset_mapping").tolist()
        model_input_keys = {"input_ids", "attention_mask"}
        inputs = {k: v.to(device) for k, v in tokenized.items() if k in model_input_keys}

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        if was_training:
            self.model.train()

        # Determine which sentences have valid word tokens and record their indices
        valid_sent_indices = []
        per_sentence_token_indices = []
        for i, (word_spans, sentence_offsets) in enumerate(zip(all_word_spans, offset_mapping)):
            seen: set[int] = set()
            word_token_indices = []
            for char_start, char_end in word_spans:
                for tok_idx, (tok_start, tok_end) in enumerate(sentence_offsets):
                    if tok_end > tok_start and tok_start < char_end and tok_end > char_start:
                        if tok_idx not in seen:
                            seen.add(tok_idx)
                            word_token_indices.append(tok_idx)
            if not word_token_indices:
                warnings.warn(
                    f"Word '{word}' was truncated away in sentence: {valid_sentences[i]!r}; skipping."
                )
                continue
            valid_sent_indices.append(i)
            per_sentence_token_indices.append(word_token_indices)

        if not valid_sent_indices:
            return {}

        # For each layer, mean-pool word tokens per sentence -> (n_valid, d_model)
        result = {}
        for layer_idx, layer_hs in enumerate(outputs.all_hidden_states):
            sentence_embs = [
                layer_hs[sent_idx][tok_indices].mean(dim=0)
                for sent_idx, tok_indices in zip(valid_sent_indices, per_sentence_token_indices)
            ]
            result[layer_idx] = torch.stack(sentence_embs)

        return result


class CosineSimilarityScorer:
    """Scores word similarity using cosine similarity of hidden-state embeddings.

    Delegates all model interaction to a ``WordEmbedder``, which caches
    embeddings so the model is run at most once per unique word, regardless of
    how many layers are scored.

    Args:
        embedder: A ``WordEmbedder`` providing per-layer embeddings.
        layer: Default layer for ``score()``. An ``int`` selects a single layer
            (negative indexing supported; defaults to -1, the last transformer
            block output). A ``(start, end)`` tuple averages the layers in
            ``all_hidden_states[start:end]``.
    """

    def __init__(self, embedder: WordEmbedder, layer: Union[int, Tuple[int, int]] = -1):
        self.embedder = embedder
        self.layer = layer

    def score(self, word1: str, word2: str) -> float:
        """Return cosine similarity for ``self.layer``.

        Args:
            word1: First word.
            word2: Second word.

        Returns:
            Cosine similarity in [-1, 1].
        """
        embs1 = self.embedder.embed_all_layers(word1)
        embs2 = self.embedder.embed_all_layers(word2)
        v1 = self._select_layer(embs1).cpu()
        v2 = self._select_layer(embs2).cpu()
        return self._cosine(v1, v2)

    def score_layers(
        self, word1: str, word2: str, layers: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """Return cosine similarity for each requested layer.

        Runs the model at most once per word regardless of how many layers are
        requested. ``layers`` defaults to all available layers.

        Args:
            word1: First word.
            word2: Second word.
            layers: Layer indices to score. Negative indices are not supported
                here; pass explicit non-negative integers. Defaults to all layers.

        Returns:
            Dict mapping layer index to cosine similarity in [-1, 1].
        """
        embs1 = self.embedder.embed_all_layers(word1)
        embs2 = self.embedder.embed_all_layers(word2)
        if layers is None:
            layers = list(embs1.keys())
        return {
            layer: self._cosine(embs1[layer].cpu(), embs2[layer].cpu())
            for layer in layers
        }

    def _select_layer(self, embs: dict[int, torch.Tensor]) -> torch.Tensor:
        """Select the embedding for ``self.layer`` from an all-layers dict."""
        n_layers = len(embs)
        if isinstance(self.layer, tuple):
            start, end = self.layer
            indices = list(range(n_layers)[start:end])
            if not indices:
                raise ValueError(
                    f"Layer range {self.layer} produced an empty slice for "
                    f"embeddings with {n_layers} layers."
                )
            return torch.stack([embs[i] for i in indices]).mean(dim=0)
        idx = self.layer if self.layer >= 0 else n_layers + self.layer
        if idx not in embs:
            raise ValueError(
                f"Layer {self.layer} (resolved to {idx}) not found in "
                f"embeddings with {n_layers} layers."
            )
        return embs[idx]

    @staticmethod
    def _cosine(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

# Provenance note (v2 release): this script produced the released file in
# data/inputs/. Re-running it is not needed to reproduce the paper; the released
# file is the canonical input. Exact re-creation also requires the miniWIT
# manifest built for the paper's BaselineLM (not released); without it, the
# script still runs but cannot apply the WIT exclusion, so its output may differ.
# Per Vulić et al. (2020), we want to gather sentences containing the SimLex-999 words from the Wikipedia dump,
# to use as context for scoring. This script gathers those sentences and saves them in a JSON file.
# Sentences in WIT are excluded.
#
# Output format: {word: [sentence1, sentence2, ...], ...}
# Up to MAX_SENTENCES_PER_WORD sentences are collected per word.

import re
import warnings
import random
import nltk
import datasets
import json
from pathlib import Path


try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


SIMLEX_PATH = Path("data/inputs/SimLex-999.txt")
WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIT_MANIFEST_PATH = Path("data/miniWIT/manifest.jsonl")  # not released; see note above
OUTPUT_PATH = Path("data/inputs/simlex_context.json")
MAX_SENTENCES_PER_WORD = 10
COLLECT_SENTENCES_PER_WORD = 50  # Collect at least this many, then sample down to MAX_SENTENCES_PER_WORD
MAX_SENTENCE_WORDS = 75  # Sentences longer than this are likely to be truncated by the model


def load_simlex_words(simlex_path: Path) -> set[str]:
    """Load the unique vocabulary of words from the SimLex-999 dataset.

    Reads word1 and word2 columns from the TSV file and returns the union
    of all unique word forms.

    Args:
        simlex_path: Path to the SimLex-999.txt TSV file.

    Returns:
        A set of lowercase word strings appearing in the SimLex-999 dataset.
    """
    words = set()
    with open(simlex_path) as f:
        header = f.readline().strip().split("\t")
        w1_idx = header.index("word1")
        w2_idx = header.index("word2")
        for line in f:
            fields = line.strip().split("\t")
            words.add(fields[w1_idx].lower())
            words.add(fields[w2_idx].lower())
    return words


def load_wit_sentences(wit_manifest_path: Path = WIT_MANIFEST_PATH) -> set[str]:
    """Load the set of sentences present in the local miniWIT manifest.

    Reads the JSONL manifest, strips the [SEP] separator token from each
    entry's text field, splits the result into sentences, and returns them
    as a set for O(1) exclusion lookup.

    Args:
        wit_manifest_path: Path to the miniWIT manifest.jsonl file.

    Returns:
        A set of sentence strings drawn from the manifest text fields.
    """
    sentences = set()
    if not Path(wit_manifest_path).exists():
        warnings.warn(
            f"WIT manifest not found at {wit_manifest_path}; continuing without WIT exclusion. "
            "Output may differ from the released file."
        )
        return sentences
    with open(wit_manifest_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            # Remove the [SEP] token used to join caption and context
            text = text.replace("[SEP]", " ")
            for sentence in extract_sentences(text):
                sentences.add(sentence)
    return sentences


def extract_sentences(article_text: str) -> list[str]:
    """Split a Wikipedia article body into individual sentences.

    Uses a simple sentence boundary detector. Handles common abbreviation
    edge cases (e.g. "U.S.", "Dr.") to reduce false splits.

    Args:
        article_text: The raw text of a single Wikipedia article.

    Returns:
        A list of sentence strings extracted from the article.
    """
    return nltk.tokenize.sent_tokenize(article_text)


def find_simlex_words_in_sentence(sentence: str, simlex_words: set[str]) -> list[str]:
    """Return all SimLex-999 words that appear in the sentence.

    Performs whole-word (token-boundary) matching so that sub-word occurrences
    (e.g. "hardship" for "hard") are not counted.

    Args:
        sentence: The sentence to search.
        simlex_words: The set of target word forms (lowercase).

    Returns:
        A list of SimLex-999 words found in the sentence (may be empty).
    """
    tokens = set(w.lower() for w in nltk.tokenize.word_tokenize(sentence))
    return [w for w in simlex_words if w.lower() in tokens]


def is_table_like(sentence: str) -> bool:
    """Return True if a sentence looks like a table row rather than prose.

    Checks for two signals common in Wikipedia table content:
    - Many embedded newlines, tabs, or pipe characters (wikitext table syntax)
    - A high ratio of non-alphabetic, non-space characters (e.g. numbers,
      pipes, dashes, and brackets that dominate in data tables)

    Args:
        sentence: The sentence to test.

    Returns:
        True if the sentence is likely table content, False otherwise.
    """
    noisy_chars = sentence.count("\n") + sentence.count("\t") + sentence.count("|")
    if noisy_chars >= 3:
        return True
    non_alpha = sum(1 for c in sentence if not c.isalpha() and not c.isspace())
    if non_alpha / len(sentence) > 0.3:
        return True
    return False


def article_contains_target_words(article_text: str, target_pattern: re.Pattern) -> bool:
    """Return True if the article text contains at least one target word.

    Used as a cheap pre-filter before sentence tokenization so that articles
    with no relevant words are skipped entirely.

    Args:
        article_text: Raw text of a Wikipedia article.
        target_pattern: Compiled regex with word-boundary alternation over all
            target words (case-insensitive).

    Returns:
        True if any target word appears in the article.
    """
    return bool(target_pattern.search(article_text))


def build_target_pattern(words: set[str]) -> re.Pattern:
    """Compile a word-boundary alternation regex for a set of words.

    Args:
        words: Set of lowercase word strings to match.

    Returns:
        A compiled, case-insensitive regex that matches any of the words at a
        word boundary.
    """
    alternation = "|".join(re.escape(w) for w in sorted(words))
    return re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)


def _collect_candidates(
    target_words: set[str],
    wit_sentences: set[str],
    collect_per_word: int,
    wikipedia_dataset_name: str,
    wikipedia_config: str,
    stall_articles: int,
    exclude_sentences: dict[str, set[str]] | None = None,
    article_filter: re.Pattern | None = None,
) -> dict[str, list[str]]:
    """Stream Wikipedia and fill per-word candidate pools.

    For each article, qualifying sentences are added to the candidate pool of
    every target word they contain. Streaming stops when all words have reached
    ``collect_per_word`` candidates or when ``stall_articles`` consecutive
    articles yield no progress.

    Args:
        target_words: Words to collect candidates for.
        wit_sentences: Sentences to exclude.
        collect_per_word: Target candidate-pool size per word.
        wikipedia_dataset_name: HuggingFace dataset identifier for Wikipedia.
        wikipedia_config: Dataset config string (language/date snapshot).
        stall_articles: Stop if this many consecutive articles yield no progress.
        exclude_sentences: If provided, sentences already present in a word's
            set are skipped and newly collected ones are added to the set to
            prevent cross-article duplicates.
        article_filter: If provided, articles that do not match this compiled
            regex are skipped before sentence tokenization.

    Returns:
        Dict mapping each target word to its list of candidate sentences.
    """
    candidates: dict[str, list[str]] = {w: [] for w in target_words}

    dataset = datasets.load_dataset(
        wikipedia_dataset_name, wikipedia_config, split="train", streaming=True
    )

    filled_count = 0
    articles_since_progress = 0

    for article_idx, article in enumerate(dataset):
        if all(len(candidates[w]) >= collect_per_word for w in target_words):
            break

        if articles_since_progress >= stall_articles:
            unfilled = [w for w in target_words if len(candidates[w]) < collect_per_word]
            print(f"  Stall detected after {stall_articles} articles without progress; stopping early.")
            print(f"  Unfilled words ({len(unfilled)}): {[(word, len(candidates[word])) for word in unfilled]}")
            break

        if article_idx % 1000 == 0:
            filled = sum(1 for w in target_words if len(candidates[w]) >= collect_per_word)
            print(f"  Articles processed: {article_idx}, words with full pools: {filled}/{len(target_words)}")

        if article_filter is not None and not article_contains_target_words(article["text"], article_filter):
            continue

        for sentence in extract_sentences(article["text"]):
            if sentence in wit_sentences:
                continue
            if not (5 < len(sentence.split()) <= MAX_SENTENCE_WORDS):
                continue
            if is_table_like(sentence):
                continue
            for word in find_simlex_words_in_sentence(sentence, target_words):
                if exclude_sentences is not None:
                    if sentence in exclude_sentences[word]:
                        continue
                    exclude_sentences[word].add(sentence)
                candidates[word].append(sentence)

        new_filled_count = sum(1 for w in target_words if len(candidates[w]) >= collect_per_word)
        if new_filled_count > filled_count:
            filled_count = new_filled_count
            articles_since_progress = 0
        else:
            articles_since_progress += 1

    return candidates


def _sample_candidates(
    word: str,
    candidates: list[str],
    needed: int,
    rng: random.Random,
) -> list[str]:
    """Sample up to ``needed`` sentences from ``candidates`` with lowercase preference.

    Partitions candidates into those where ``word`` appears in lowercase (i.e.
    not as a proper noun) and the rest, then fills the quota from the lowercase
    pool first before drawing from the remainder.

    Args:
        word: The target word (used for case-sensitive whole-word matching).
        candidates: Pool of candidate sentences to sample from.
        needed: Maximum number of sentences to return.
        rng: Random instance for reproducible sampling.

    Returns:
        A list of up to ``needed`` sampled sentences.
    """
    pattern = re.compile(r"\b" + re.escape(word) + r"\b")
    lowercase_sents = [s for s in candidates if pattern.search(s)]
    other_sents = [s for s in candidates if not pattern.search(s)]
    sampled = rng.sample(lowercase_sents, min(needed, len(lowercase_sents)))
    remaining = needed - len(sampled)
    if remaining > 0:
        sampled += rng.sample(other_sents, min(remaining, len(other_sents)))
    return sampled


def gather_context_sentences(
    simlex_words: set[str],
    wit_sentences: set[str],
    collect_per_word: int = COLLECT_SENTENCES_PER_WORD,
    sample_per_word: int = MAX_SENTENCES_PER_WORD,
    wikipedia_dataset_name: str = WIKIPEDIA_DATASET,
    wikipedia_config: str = WIKIPEDIA_CONFIG,
    seed: int = 42,
    stall_articles: int = 100_000,
) -> dict[str, list[str]]:
    """Stream Wikipedia and collect context sentences for each SimLex-999 word.

    For each article, sentences are extracted and checked for SimLex-999 words.
    A sentence is added to a word's candidate pool if it is not present in WIT
    and that word has not yet reached ``collect_per_word`` candidates. Streaming
    stops early once every word has filled its candidate pool. If no new word
    fills its pool within ``stall_articles`` consecutive articles, streaming
    stops early and the best available candidates are used. Finally,
    ``sample_per_word`` sentences are drawn at random from each word's pool.

    Args:
        simlex_words: Set of word forms to collect sentences for.
        wit_sentences: Set of sentences to exclude.
        collect_per_word: Minimum number of candidate sentences to collect per word
            before sampling.
        sample_per_word: Number of sentences to randomly sample from each
            word's candidate pool for the final output.
        wikipedia_dataset_name: HuggingFace dataset identifier for Wikipedia.
        wikipedia_config: Dataset config string (language/date snapshot).
        seed: Random seed for reproducible sampling.
        stall_articles: Stop streaming if this many consecutive articles pass
            without any new word reaching its pool quota.

    Returns:
        A dict mapping each SimLex-999 word to a randomly sampled list of
        ``sample_per_word`` sentences (or fewer if not enough were found).
    """
    rng = random.Random(seed)
    candidates = _collect_candidates(
        simlex_words, wit_sentences, collect_per_word,
        wikipedia_dataset_name, wikipedia_config, stall_articles,
    )
    return {w: _sample_candidates(w, sents, sample_per_word, rng) for w, sents in candidates.items()}


def update_context_sentences(
    existing: dict[str, list[str]],
    wit_sentences: set[str],
    collect_per_word: int = COLLECT_SENTENCES_PER_WORD,
    sample_per_word: int = MAX_SENTENCES_PER_WORD,
    wikipedia_dataset_name: str = WIKIPEDIA_DATASET,
    wikipedia_config: str = WIKIPEDIA_CONFIG,
    seed: int = 42,
    stall_articles: int = 100_000,
) -> dict[str, list[str]]:
    """Fill gaps in an existing context-sentences dict by streaming Wikipedia.

    Words that already have ``sample_per_word`` sentences are left untouched.
    For underfilled words, Wikipedia is streamed and each article is first
    checked with a cheap regex pre-filter; only articles that contain at least
    one target word are sentence-tokenized. New sentences (not already present
    for a word) are collected into per-word candidate pools. Streaming stops
    when every underfilled word has ``collect_per_word`` new candidates, or
    when no progress has been made for ``stall_articles`` consecutive articles.
    Finally, newly sampled sentences are appended to the existing lists up to
    ``sample_per_word``.

    Args:
        existing: Dict mapping SimLex-999 words to their current sentence lists.
        wit_sentences: Set of sentences to exclude.
        collect_per_word: Target number of new candidate sentences per word
            before sampling.
        sample_per_word: Maximum sentences per word in the final output.
        wikipedia_dataset_name: HuggingFace dataset identifier for Wikipedia.
        wikipedia_config: Dataset config string (language/date snapshot).
        seed: Random seed for reproducible sampling.
        stall_articles: Stop if this many consecutive articles yield no progress.

    Returns:
        An updated dict with each word's list extended up to ``sample_per_word``.
    """
    needs_sentences = {w for w, sents in existing.items() if len(sents) < sample_per_word}
    if not needs_sentences:
        print("  All words already have enough sentences. Nothing to do.")
        return dict(existing)

    print(f"  {len(needs_sentences)} words need more sentences.")

    exclude_sentences = {w: set(existing.get(w, [])) for w in needs_sentences}
    new_candidates = _collect_candidates(
        needs_sentences, wit_sentences, collect_per_word,
        wikipedia_dataset_name, wikipedia_config, stall_articles,
        exclude_sentences=exclude_sentences,
        article_filter=build_target_pattern(needs_sentences),
    )

    rng = random.Random(seed)
    result = dict(existing)
    for word in needs_sentences:
        needed = sample_per_word - len(existing.get(word, []))
        new = _sample_candidates(word, new_candidates[word], needed, rng)
        result[word] = existing.get(word, []) + new
    return result


def save_context_sentences(word_to_sentences: dict[str, list[str]], output_path: Path) -> None:
    """Write the word-to-sentences mapping to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        word_to_sentences: Dict mapping SimLex-999 words to lists of sentences,
            as returned by ``gather_context_sentences``.
        output_path: Destination path for the output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(word_to_sentences, f, indent=2)


def main() -> None:
    """Entry point: load SimLex words and WIT sentences, stream Wikipedia,
    and save qualifying context sentences.

    Modes
    -----
    default : Stream Wikipedia from scratch and build the full sentence dict.
    --update: Load an existing JSON, identify underfilled words, and fill them.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Gather SimLex-999 context sentences from Wikipedia."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fill words with fewer than MAX_SENTENCES_PER_WORD sentences in an existing JSON.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_PATH,
        help="Existing JSON to update (only used with --update). Defaults to OUTPUT_PATH.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination path for the output JSON. Defaults to OUTPUT_PATH.",
    )
    parser.add_argument(
        "--stall-articles",
        type=int,
        default=100_000,
        metavar="N",
        help="Stop streaming if N consecutive articles yield no progress (default: 100000).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=COLLECT_SENTENCES_PER_WORD,
        metavar="N",
        help=f"Candidate pool size to collect per word before sampling (default: {COLLECT_SENTENCES_PER_WORD}).",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=MAX_SENTENCES_PER_WORD,
        metavar="N",
        help=f"Maximum sentences per word in the final output (default: {MAX_SENTENCES_PER_WORD}).",
    )
    args = parser.parse_args()

    print("Loading SimLex-999 words...")
    simlex_words = load_simlex_words(SIMLEX_PATH)
    print(f"  {len(simlex_words)} unique words")

    print(f"Loading WIT sentences to exclude from {WIT_MANIFEST_PATH}...")
    wit_sentences = load_wit_sentences()
    print(f"  {len(wit_sentences)} sentences")

    if args.update:
        print(f"Loading existing context sentences from {args.input}...")
        with open(args.input) as f:
            existing = json.load(f)
        print(f"  {len(existing)} words loaded")
        print("Streaming Wikipedia and updating context sentences...")
        word_to_sentences = update_context_sentences(
            existing, wit_sentences,
            collect_per_word=args.pool_size,
            sample_per_word=args.max_sentences,
            stall_articles=args.stall_articles,
        )
    else:
        print("Streaming Wikipedia and gathering context sentences...")
        word_to_sentences = gather_context_sentences(
            simlex_words, wit_sentences,
            collect_per_word=args.pool_size,
            sample_per_word=args.max_sentences,
            stall_articles=args.stall_articles,
        )

    print(f"Saving to {args.output}...")
    save_context_sentences(word_to_sentences, args.output)
    print("Done.")


if __name__ == "__main__":
    main()

# Provenance note (v2 release): this script produced the released file in
# data/inputs/. Re-running it is not needed to reproduce the paper; the released
# file is the canonical input. Exact re-creation also requires the miniWIT
# manifest built for the paper's BaselineLM (not released); without it, the
# script still runs but cannot apply the WIT exclusion, so its output may differ.
# Script to gather corpus sentences to calculuate \mu and \sigma for standardization in the SimLex-999 experiments.
# We create a list of sentences from Wikipedia, excluding any sentences that appear in the WIT manifest. 
#
# Output format: one sentence per line. 

import warnings
import nltk
import datasets
import json
from pathlib import Path


try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIT_MANIFEST_PATH = Path("data/miniWIT/manifest.jsonl")  # not released; see note above
OUTPUT_PATH = Path("data/inputs/corpus.txt")
TARGET_WORDS = 500_000


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


def collect_sentences(
    target_word_count: int,
    wit_sentences: set[str],
    wikipedia_dataset_name: str,
    wikipedia_config: str,
) -> list[str]:
    """
    Stream Wikipedia and collect sentences until the target word count is reached.

    Args:
        target_word_count: The number of words to collect sentences for.
        wit_sentences: Sentences to exclude.
        wikipedia_dataset_name: HuggingFace dataset identifier for Wikipedia.
        wikipedia_config: Dataset config string (language/date snapshot).
        exclude_sentences: If provided, sentences already present in a word's
            set are skipped and newly collected ones are added to the set to
            prevent cross-article duplicates.
        wikipedia_dataset_name: HuggingFace dataset identifier for Wikipedia.
        wikipedia_config: Dataset config string (language/date snapshot).
        

    Returns:
        List of unique sentences, excluding those in WIT.
    """
    all_sentences = set()
    dataset = datasets.load_dataset(
        wikipedia_dataset_name, wikipedia_config, split="train", streaming=True
    )

    num_words = 0

    for article_idx, article in enumerate(dataset):
        if num_words >= target_word_count:
            break


        if article_idx % 1000 == 0:
            print(f"  Articles processed: {article_idx}, words collected: {num_words}")

        for sentence in extract_sentences(article["text"]):
            if sentence in wit_sentences:
                continue
            if is_table_like(sentence):
                continue
            sentence = sentence.strip().replace("\n", " ")
            sentence_length = len(sentence.split())
            if sentence_length < 5:
                continue
            if sentence not in all_sentences:
                all_sentences.add(sentence)
                num_words += sentence_length

    print(f"Finished collecting sentences. Total articles processed: {article_idx}, total words collected: {num_words}, total sentences collected: {len(all_sentences)}")

    return list(all_sentences)


def save_sentences(sentences: list[str], output_path: Path) -> None:
    """Write a list of sentences to a TXT file, one per line.

    Creates parent directories if they do not exist.

    Args:
        sentences: List of sentence strings to save.
        output_path: Destination path for the output TXT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


def main() -> None:
    """Entry point: stream Wikipedia and save qualifying sentences.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Gather a corpus of sentences from Wikipedia."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination path for the output TXT file. Defaults to OUTPUT_PATH.",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=TARGET_WORDS,
        metavar="N",
        help=f"Target number of words to collect sentences for (default: {TARGET_WORDS}).",
    )
    args = parser.parse_args()

    print(f"Loading WIT sentences to exclude from {WIT_MANIFEST_PATH}...")
    wit_sentences = load_wit_sentences()
    print(f"  {len(wit_sentences)} sentences")

    print("Streaming Wikipedia and gathering sentences...")
    sentences = collect_sentences(
        target_word_count=args.target_words,
        wit_sentences=wit_sentences,
        wikipedia_dataset_name=WIKIPEDIA_DATASET,
        wikipedia_config=WIKIPEDIA_CONFIG
    )

    print(f"Saving to {args.output}...")
    save_sentences(sentences, args.output)
    print("Done.")


if __name__ == "__main__":
    main()

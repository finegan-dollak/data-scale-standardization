# Inputs for the generation scripts

| File | Contents |
|---|---|
| `simlex_context.json` | Up to 10 English Wikipedia sentences for each SimLex-999 word, as `{word: [sentence, ...]}`. Word vectors in the paper are averaged over these sentences. |
| `corpus.txt` | About 500,000 words of English Wikipedia, one sentence per line. Used to fit standardization and to sample token pairs for anisotropy and rogue dimensionality. |
| `SimLex-999.txt` | Not included. Download it yourself; see `generation/README.md`. |

Both included files were drawn from the `wikimedia/wikipedia` dataset (config `20231101.en`) by `generation/gather_context_sentences.py` and `generation/gather_corpus.py`. See `generation/README.md` for the sampling details.

## License

Unlike the rest of `data/`, these two files contain text from Wikipedia and are therefore licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/. The text is from English Wikipedia (https://en.wikipedia.org), written by Wikipedia contributors, via the Wikimedia Foundation's `wikimedia/wikipedia` dataset on Hugging Face.

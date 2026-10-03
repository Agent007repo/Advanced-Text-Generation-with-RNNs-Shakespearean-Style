# Recurrent Text Generation in Shakespearean Style

A PyTorch notebook for autoregressive subword text generation. A configuration selects vanilla RNN, LSTM, or GRU, BPE or WordPiece tokenization, model size, gradient clipping, and decoding settings. One execution trains the configured model; a controlled multi-model benchmark requires separate runs.

## Run

Use Python 3.11+, install `requirements.txt`, and place UTF-8 `shakespeare.txt` in this directory. The notebook optionally supports a Colab upload if the file is absent. Execute `Prof_Faith_Text_Generation_with_RNNs.ipynb` from the repository root.

```bash
pip install -r requirements.txt
jupyter notebook Prof_Faith_Text_Generation_with_RNNs.ipynb
```

The raw corpus is split into disjoint training and validation text before tokenizer training. Tokenizer caches are keyed by the training corpus digest. Sliding windows are constructed lazily within each partition, and each independent minibatch starts with a fresh recurrent state. Unidirectional models preserve causal next-token prediction; bidirectional configuration is rejected.

Temperature sampling uses stable probabilities and requires a finite positive temperature. Beam search and sampling consume the prompt exactly once. Evaluation reports validation cross-entropy and perplexity, with qualitative generated samples. Optional stemming requires NLTK resources; when unavailable the notebook falls back to unstemmed text. Install the resources explicitly if stemming is needed.

## Evaluation status

Old notebook outputs were removed because overlapping randomly divided windows leaked corpus context across training and validation. No corrected real-corpus perplexity, architecture comparison, energy consumption, or inference-efficiency benchmark has been reproduced. These are implementation experiments, not a validated language-model benchmark.

```bash
python -m unittest discover -s tests -p test_regressions.py -v
```

Six regression tests train and evaluate all three small recurrent models, exercise a partial final batch, check lazy targets and invalid configurations, and verify stable decoding and prompt handling. They use a synthetic tokenizer and do not replace a full corpus run.

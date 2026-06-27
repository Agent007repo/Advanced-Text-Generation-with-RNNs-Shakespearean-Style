# Shakespearean Text Generation with RNNs

Educational sequence-modeling notebook exploring neural text generation in a Shakespeare-style corpus.

## Project Maturity

Notebook learning project. This repository is useful as evidence of sequence-modeling exposure, but it should not be presented as a production NLP system.

## What This Project Demonstrates

- Preparing text data for sequence modeling.
- Training recurrent neural-network style models for next-token generation.
- Generating text samples from learned sequence patterns.
- Understanding the limitations of older RNN-style architectures compared with modern transformer-based language models.

## Main Artifact

- `Prof_Faith_Text_Generation_with_RNNs.ipynb`: notebook containing the modeling workflow and generated outputs.

## Tech Stack

- Python
- Jupyter Notebook
- PyTorch
- NumPy
- Tokenizers / NLTK-style preprocessing utilities
- Matplotlib / scikit-learn utilities where applicable

## Local Setup

```bash
git clone https://github.com/Agent007repo/Advanced-Text-Generation-with-RNNs-Shakespearean-Style.git
cd Advanced-Text-Generation-with-RNNs-Shakespearean-Style
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Prof_Faith_Text_Generation_with_RNNs.ipynb
```

## Reviewer Notes

This project should be read as an educational NLP artifact. For recruiter screening, it supports basic familiarity with neural sequence modeling. For ML/AI engineering roles, stronger evidence should come from the more applied repositories such as `autonomous-pm-engine` and `SCRI-Supply-Chain-Risk-Intelligence-System`.

## Recommended Next Improvements

- Add a clean comparison table for RNN, LSTM, and GRU runs.
- Add loss/perplexity plots.
- Add generated sample text for multiple temperatures.
- Split preprocessing, training, and generation into reusable scripts.
- Add a short note comparing RNN-based generation with transformer-based LLMs.

# Advanced Text Generation with RNNs: Shakespearean Style

This project compares recurrent neural-network architectures for character or token-level text generation using Shakespeare-style text. It demonstrates sequence modeling fundamentals, training stability techniques, and controlled generation strategies.

## What This Shows

- Deep learning fundamentals for sequence data
- RNN, LSTM, and GRU architecture comparison
- Embeddings, bidirectional layers, gradient clipping, and teacher forcing
- Temperature sampling and beam-search style generation
- Evaluation using perplexity plus qualitative output review

## Repository Contents

| File | Purpose |
|---|---|
| `Prof_Faith_Text_Generation_with_RNNs.ipynb` | End-to-end notebook for preprocessing, model training, generation, and evaluation |
| `README.md` | Project overview and reproducibility guide |

## Methodology

1. Prepare the Shakespeare text corpus for sequence modeling.
2. Build comparable recurrent models: vanilla RNN, LSTM, and GRU.
3. Train models with optimization safeguards such as gradient clipping.
4. Generate text under different decoding settings to compare creativity and coherence.
5. Evaluate model behavior with perplexity and qualitative inspection of generated samples.

## Tech Stack

- Python
- Jupyter Notebook
- TensorFlow or Keras-style recurrent models, as implemented in the notebook
- NumPy and standard data-processing utilities

## How To Run

```bash
git clone https://github.com/Agent007repo/Advanced-Text-Generation-with-RNNs-Shakespearean-Style.git
cd Advanced-Text-Generation-with-RNNs-Shakespearean-Style
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Prof_Faith_Text_Generation_with_RNNs.ipynb
```

## Recruiter Signal

This is a learning-oriented deep learning project. Its value is not production deployment; it shows comfort with sequence models, model comparison, and the tradeoff between generative diversity and coherence.

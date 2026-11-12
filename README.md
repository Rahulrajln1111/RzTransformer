# RzTransformer

A from-scratch Transformer (encoder–decoder) trained for **English → Hinglish** translation,
built with PyTorch. The model, training loop, BLEU evaluation, and interactive inference
are all included.

## Project layout

```
.
├── model.py               # Transformer architecture (embeddings, MHA, FFN, encoder/decoder)
├── train.py               # Training pipeline (tokenization, data loaders, optimization, checkpoints)
├── evaluate_bleu.py       # Corpus BLEU / chrF evaluation on a validation split
├── test_transformer.py    # Interactive greedy-decoding translator
├── requirements.txt
└── tokenizer/             # WordLevel tokenizers (built on first training run)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU (CUDA) is recommended; training falls back to CPU otherwise.

## Training

```bash
python train.py
```

Common overrides:

```bash
python train.py --minutes 30 --batch-size 64 --d-model 512 --layers 6
```

Checkpoints are saved to `checkpoints/best_transformer.pt`. Tokenizers are generated on the
first run and cached under `tokenizer/`.

## Evaluation (BLEU)

```bash
python evaluate_bleu.py --checkpoint checkpoints/best_transformer.pt --n 500
```

Multiple checkpoints can be compared in one run:

```bash
python evaluate_bleu.py --checkpoint a.pt b.pt --n 500
```

## Inference

```bash
python test_transformer.py
# or with an explicit checkpoint:
python test_transformer.py --checkpoint checkpoints/best_transformer.pt
```

Then type an English sentence at the prompt; the model replies with Hinglish.

## Dataset

Training uses `findnitai/english-to-hinglish` (Hugging Face). It is downloaded automatically
on the first run.

## Notes

- Model weights are not committed (see `.gitignore`); checkpoints live under `checkpoints/`.
- `checkpoints/` and `.vscode/` are gitignored.

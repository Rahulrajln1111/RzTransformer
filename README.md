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

## Results

Corpus-level scores on 500 validation sentences with greedy decoding, measured with
`sacrebleu` (BLEU with the default `13a` tokenizer, plus chrF):

| Checkpoint | Params | Epochs | Train budget | Val loss | BLEU | chrF | Copy-src | Empty |
|---|---|---|---|---|---|---|---|---|
| `model_d384_15min.pt` | 38.5M | 6 | 15 min | 2.6803 | 43.03 | 61.93 | 0.0% | 1.8% |
| `best_transformer.pt` | 77.5M | 8 | 45 min | 2.4246 | **49.80** | **67.64** | 0.0% | 0.2% |

**Copy-src** is the fraction of outputs that are byte-identical to the source sentence and
**empty** the fraction of blank outputs — both are degenerate-behaviour diagnostics that raw
BLEU hides. Neither model is copying the input, and the larger model almost never emits
an empty string.

Hardware: RTX 3050 Laptop (6 GB), bf16 autocast, batch size 128, sequence length 32. Model
weights are not committed; the numbers above come from local checkpoints.

Reproduce with:

```bash
python evaluate_bleu.py --checkpoint checkpoints/model_d384_15min.pt checkpoints/best_transformer.pt --n 500
```

## Methodology

**Data.** `findnitai/english-to-hinglish`, 189,102 sentence pairs. A 95/5 split with a fixed
seed (42) gives 179,646 train / 9,456 validation pairs; evaluation uses the first 500
validation sentences. The same split is used for checkpoint selection and for reporting, so
these numbers are validation scores, not test scores.

**Tokenization.** Word-level tokenizers (whitespace pre-tokenization) trained on the training
split only: 19,289 source tokens and 22,935 target tokens. Sequence length 32 was chosen from
the length distribution (mean 9 tokens, p99 = 28) rather than guessed.

**Architecture.** Encoder–decoder Transformer, sinusoidal positional encoding, post-norm
residual blocks (`LayerNorm` after each sublayer), `d_ff = 4 x d_model`, dropout 0.1, and a
separate projection head that returns raw logits. No weight tying.

**Optimization.** AdamW (fused), lr 3e-4, 400 warmup steps followed by inverse-square-root
decay, gradient clipping at 1.0, label smoothing 0.1, bf16 autocast.

**Budget.** Training is wall-clock budgeted (`--minutes`) instead of fixed-epoch; the run stops
when the budget is exhausted and the best validation checkpoint is kept. The two checkpoints
above are 15-minute and 45-minute runs of the same pipeline.

**Decoding and metrics.** Batched greedy decoding. BLEU and chrF are computed on detokenized
text; copy-source and empty-output rates are reported alongside them as sanity checks.

## Throughput

Compute-only training step (forward + backward + AdamW update) with synthetic batches — no
data loading, so this isolates the model and optimizer. 40 steps timed after 10 warmup steps,
batch 128 x seq 32 (4,096 tokens/step), bf16, RTX 3050 Laptop 6 GB:

| Config | Params | ms/step | samples/s | tokens/s | Peak GPU mem |
|---|---|---|---|---|---|
| d_model 384, N=4, h=6, d_ff 1024 | 38.5M | 161.6 | 792 | ~25.3k | 2.46 GB |
| d_model 512, N=6, h=8, d_ff 2048 | 77.5M | 314.4 | 407 | ~13.0k | 3.78 GB |

What produced these numbers:

- `F.scaled_dot_product_attention` with a boolean mask instead of a hand-written
  scores/softmax/weights path, and `F.layer_norm` instead of a custom `LayerNorm` module.
- Cross-entropy applied directly to the projection head's logits (no `log_softmax` + `NLLLoss`).
- Dataset pre-tokenized to integer tensors once, instead of re-tokenizing per item per epoch.
- Attention masks built once per batch in the collate function, not stored per row.
- Fused AdamW, bf16 autocast, and `set_to_none=True` on `zero_grad`.
- `torch.compile` was benchmarked and left off: no measurable gain at this size.

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

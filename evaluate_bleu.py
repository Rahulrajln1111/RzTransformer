"""
Corpus BLEU for one or more checkpoints on the same validation split.

Usage:
    python evaluate_bleu.py --checkpoint checkpoints/model_d384_15min.pt
    python evaluate_bleu.py --checkpoint a.pt b.pt --n 500

Every checkpoint sees exactly the same source sentences and references, so the
scores are comparable. Greedy decoding, batched.
"""

import argparse
import time

import torch
import sacrebleu
from tokenizers import Tokenizer
from datasets import load_dataset

from model import buil_transformer
import train as T


def load_model(path, device, fallback_config=None):

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    config = checkpoint.get("model_config") or fallback_config

    if config is None:
        raise SystemExit(
            f"{path} has no model_config (trained before checkpoints recorded "
            f"their shape). Pass --legacy-shape to supply it."
        )

    model = buil_transformer(**config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config, checkpoint


@torch.inference_mode()
def translate_batch(model, src, src_mask, tgt_sos, tgt_eos, tgt_pad, max_len):
    """Greedy decode a batch; returns the generated ids including SOS."""

    encoder_output = model.encode(src, src_mask)

    generated = torch.full(
        (src.size(0), 1), tgt_sos, dtype=torch.long, device=src.device
    )
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):

        length = generated.size(1)

        causal_mask = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=src.device)
        )
        decoder_mask = (
            causal_mask.unsqueeze(0)
            & (generated != tgt_pad).unsqueeze(1).unsqueeze(2)
        )

        logits = model.project(
            model.decode(encoder_output, src_mask, generated, decoder_mask)
        )

        next_token = logits[:, -1, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        finished |= next_token == tgt_eos

        if finished.all():
            break

    return generated


def evaluate(path, val_split, tokenizer_src, tokenizer_tgt, device,
             n_samples, batch_size, fallback_config=None):

    model, config, checkpoint = load_model(path, device, fallback_config)

    seq_len = config["src_seq_len"]

    # Same split, same seed, same first `n_samples` rows for every checkpoint.
    subset = val_split.select(range(min(n_samples, len(val_split))))

    dataset, _ = T.tokenize_split(
        subset,
        tokenizer_src,
        tokenizer_tgt,
        {"seq_len": seq_len, "src_lang": "en", "tgt_lang": "hi_ng"},
    )

    references = [item["hi_ng"] for item in subset["translation"]]

    _, _, src_pad = T.get_special_tokens(tokenizer_src, "source")
    tgt_sos, tgt_eos, tgt_pad = T.get_special_tokens(tokenizer_tgt, "target")

    hypotheses = []
    started = time.time()

    for start in range(0, len(dataset), batch_size):

        stop = min(start + batch_size, len(dataset))
        src = dataset.src[start:stop].to(device)

        src_mask = (src != src_pad).unsqueeze(1).unsqueeze(2)

        generated = translate_batch(
            model, src, src_mask, tgt_sos, tgt_eos, tgt_pad, seq_len
        )

        generated = generated[:, 1:].tolist()

        for ids in generated:
            if tgt_eos in ids:
                ids = ids[:ids.index(tgt_eos)]
            hypotheses.append(
                tokenizer_tgt.decode(ids, skip_special_tokens=True)
            )

    elapsed = time.time() - started

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    # Copying the English source is the main failure mode of this model,
    # so report how often it happens.
    sources = [item["en"] for item in subset["translation"]]
    copy_rate = sum(
        h.strip().lower() == s.strip().lower()
        for h, s in zip(hypotheses, sources)
    ) / max(len(sources), 1)

    empty_rate = sum(not h.strip() for h in hypotheses) / max(len(hypotheses), 1)

    return {
        "path": path,
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "bleu": bleu.score,
        "chrf": chrf.score,
        "copy_rate": copy_rate,
        "empty_rate": empty_rate,
        "seconds": elapsed,
        "params": sum(p.numel() for p in model.parameters()) / 1e6,
        "hypotheses": hypotheses,
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--show", type=int, default=5,
                        help="print this many example translations per model")
    parser.add_argument(
        "--legacy-shape", action="store_true",
        help="checkpoint has no model_config: assume d512 N6 seq64 (the original run)",
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    fallback_config = None

    if args.legacy_shape:
        fallback_config = {
            "src_vocab_sz": 19289,
            "tgt_vocab_sz": 22935,
            "src_seq_len": args.seq_len,
            "tgt_seq_len": args.seq_len,
            "d_model": args.d_model,
            "N": args.layers,
            "num_head": args.heads,
            "dropout": 0.1,
            "d_ff": args.d_ff,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer_src = Tokenizer.from_file(T.CONFIG["tokenizer_dir"] + "/tokenizer_en.json")
    tokenizer_tgt = Tokenizer.from_file(T.CONFIG["tokenizer_dir"] + "/tokenizer_hi_ng.json")

    val_split = load_dataset(T.CONFIG["dataset_name"])["train"].train_test_split(
        test_size=T.CONFIG["validation_split"], seed=T.CONFIG["seed"]
    )["test"]

    print(f"Validation sentences: {min(args.n, len(val_split))}")
    print(f"Device: {device}\n")

    results = []

    for path in args.checkpoint:
        print(f"Evaluating {path} ...")
        result = evaluate(
            path, val_split, tokenizer_src, tokenizer_tgt, device,
            args.n, args.batch_size, fallback_config,
        )
        results.append(result)
        print(
            f"  BLEU {result['bleu']:.2f} | chrF {result['chrf']:.2f} "
            f"| copy-src {result['copy_rate'] * 100:.1f}% "
            f"| empty {result['empty_rate'] * 100:.1f}% "
            f"| {result['seconds']:.0f}s\n"
        )

    print("=" * 78)
    print(f"{'checkpoint':44s} {'BLEU':>6s} {'chrF':>6s} {'copy%':>6s} {'params':>7s}")
    print("-" * 78)

    for result in results:
        print(
            f"{result['path']:44s} {result['bleu']:6.2f} {result['chrf']:6.2f} "
            f"{result['copy_rate'] * 100:6.1f} {result['params']:6.1f}M"
        )

    print("=" * 78)

    if args.show:

        subset = val_split.select(range(min(args.show, len(val_split))))
        sources = [item["en"] for item in subset["translation"]]
        references = [item["hi_ng"] for item in subset["translation"]]

        for result in results:
            print(f"\n--- {result['path']} ---")
            for src, ref, hyp in zip(
                sources, references, result["hypotheses"][:args.show]
            ):
                print(f"  src: {src}")
                print(f"  ref: {ref}")
                print(f"  out: {hyp}")


if __name__ == "__main__":
    main()

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader

from model import buil_transformer


CONFIG = {
    "dataset_name": "findnitai/english-to-hinglish",

    "src_lang": "en",
    "tgt_lang": "hi_ng",

    "tokenizer_dir": "tokenizer",

    # 99.5% of this dataset fits in 30 tokens (p99 = 28), so 32 is enough.
    "seq_len": 32,
    "batch_size": 128,

    "num_workers": 4,
    "validation_split": 0.05,
    "seed": 42,

    # Model
    "d_model": 384,
    "N": 4,
    "num_head": 6,
    "d_ff": 1024,
    "dropout": 0.1,

    # Optimisation
    "epochs": 30,
    "lr": 3e-4,
    "warmup_steps": 400,
    "grad_clip": 1.0,
    "label_smoothing": 0.1,
    "precision": "bf16",          # "bf16" | "fp32"
    "compile": False,             # torch.compile: no gain at this size (measured)

    # Stop training after this many minutes of pure compute.
    "time_budget_min": 15.0,
    "max_val_batches": 80,
    "log_every": 50,

    "checkpoint_dir": "checkpoints",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--precision", choices=["bf16", "fp32"], default=None)
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-time-budget", action="store_true")
    args = p.parse_args()

    for arg, key in [
        ("minutes", "time_budget_min"),
        ("batch_size", "batch_size"),
        ("seq_len", "seq_len"),
        ("d_model", "d_model"),
        ("layers", "N"),
        ("heads", "num_head"),
        ("d_ff", "d_ff"),
        ("lr", "lr"),
        ("precision", "precision"),
        ("compile", "compile"),
        ("workers", "num_workers"),
    ]:
        value = getattr(args, arg)
        if value is not None:
            CONFIG[key] = value

    if args.no_time_budget:
        CONFIG["time_budget_min"] = float("inf")

    return CONFIG


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):

    tokenizer_dir = Path(config["tokenizer_dir"])
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = tokenizer_dir / f"tokenizer_{lang}.json"

    if not tokenizer_path.exists():

        print(f"Building tokenizer for {lang}...")

        tokenizer = Tokenizer(
            WordLevel(unk_token="[UNK]")
        )

        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=[
                "[UNK]",
                "[PAD]",
                "[SOS]",
                "[EOS]"
            ],
            min_frequency=2
        )

        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer=trainer
        )

        tokenizer.save(str(tokenizer_path))

        print(f"Tokenizer saved: {tokenizer_path}")

    else:

        print(f"Loading tokenizer: {tokenizer_path}")

        tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

    return tokenizer


def get_special_tokens(tokenizer, name):
    """Fetch SOS/EOS/PAD ids and fail loudly if the vocabulary is incomplete."""
    sos = tokenizer.token_to_id("[SOS]")
    eos = tokenizer.token_to_id("[EOS]")
    pad = tokenizer.token_to_id("[PAD]")

    assert None not in [sos, eos, pad], (
        f"{name} tokenizer is missing a required special token."
    )

    return sos, eos, pad


class TranslationDataset(torch.utils.data.Dataset):
    """
    Holds pre-tokenised, pre-padded id tensors.

    Tokenising and building masks per item (the original approach) meant
    ~378k Python-level encode() calls and ~189k `tril` allocations per epoch,
    which starved the GPU. Everything static is now computed once, and the
    DataLoader only has to gather rows.
    """

    def __init__(self, src, tgt_in, label):
        self.src = src
        self.tgt_in = tgt_in
        self.label = label

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, index):
        return index


def pad_rows(rows, seq_len, pad_id, sos_id, eos_id):
    """Right-pad `[SOS] + ids + [EOS]` to seq_len in one preallocated tensor."""
    out = torch.full((len(rows), seq_len), pad_id, dtype=torch.long)

    for i, ids in enumerate(rows):
        ids = ids[:seq_len - 2]
        out[i, 0] = sos_id
        out[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
        out[i, 1 + len(ids)] = eos_id

    return out


def tokenize_split(split, tokenizer_src, tokenizer_tgt, config):

    src_texts = [item[config["src_lang"]] for item in split["translation"]]
    tgt_texts = [item[config["tgt_lang"]] for item in split["translation"]]

    # encode_batch is the fast path (Rust, multi-threaded) over per-item encode().
    src_ids = tokenizer_src.encode_batch(src_texts)
    tgt_ids = tokenizer_tgt.encode_batch(tgt_texts)

    src_sos, src_eos, src_pad = get_special_tokens(tokenizer_src, "source")
    tgt_sos, tgt_eos, tgt_pad = get_special_tokens(tokenizer_tgt, "target")

    seq_len = config["seq_len"]

    src = pad_rows([e.ids for e in src_ids], seq_len, src_pad, src_sos, src_eos)
    tgt = pad_rows([e.ids for e in tgt_ids], seq_len, tgt_pad, tgt_sos, tgt_eos)

    # Decoder gets everything except the last token, labels everything except the first.
    tgt_in = tgt[:, :-1].contiguous()
    label = tgt[:, 1:].contiguous()
    label[label == tgt_pad] = -100

    return TranslationDataset(src, tgt_in, label), tgt_pad


def make_collate_fn(dataset, src_pad, tgt_pad, causal_mask):
    """Build masks for a whole batch on the fly instead of storing one per row."""

    def collate_fn(indices):
        idx = torch.as_tensor(indices, dtype=torch.long)

        src = dataset.src.index_select(0, idx)
        tgt_in = dataset.tgt_in.index_select(0, idx)
        label = dataset.label.index_select(0, idx)

        # (B, 1, 1, S) - False marks padding, which must not be attended to.
        encoder_mask = (src != src_pad).unsqueeze(1).unsqueeze(2)

        # (B, 1, S-1, S-1) - causal mask AND padding mask, broadcast together.
        decoder_mask = causal_mask.unsqueeze(0) & (tgt_in != tgt_pad).unsqueeze(1).unsqueeze(2)

        return {
            "encoder_input": src,
            "decoder_input": tgt_in,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
        }

    return collate_fn


def get_ds(config):

    print("Loading dataset...")

    ds_raw = load_dataset(
        config["dataset_name"]
    )

    # This dataset provides a train split
    raw_train = ds_raw["train"]

    # Split into training and validation
    split = raw_train.train_test_split(
        test_size=config["validation_split"],
        seed=config["seed"]
    )

    train_raw = split["train"]
    val_raw = split["test"]

    print("Training examples:", len(train_raw))
    print("Validation examples:", len(val_raw))

    # Build tokenizers using training data only
    tokenizer_src = get_or_build_tokenizer(
        config,
        train_raw,
        config["src_lang"]
    )

    tokenizer_tgt = get_or_build_tokenizer(
        config,
        train_raw,
        config["tgt_lang"]
    )

    print("English vocabulary:", tokenizer_src.get_vocab_size())
    print("Hinglish vocabulary:", tokenizer_tgt.get_vocab_size())

    return (
        train_raw,
        val_raw,
        tokenizer_src,
        tokenizer_tgt
    )


def create_dataloaders(config):

    (train_raw, val_raw, tokenizer_src, tokenizer_tgt) = get_ds(config)

    t0 = time.time()

    print("Tokenising dataset once...")

    train_dataset, tgt_pad = tokenize_split(
        train_raw, tokenizer_src, tokenizer_tgt, config
    )
    val_dataset, _ = tokenize_split(
        val_raw, tokenizer_src, tokenizer_tgt, config
    )

    print(f"Tokenised in {time.time() - t0:.1f}s")

    # Static causal mask, shared by every batch.
    seq_len = config["seq_len"]
    causal_mask = torch.tril(
        torch.ones(seq_len - 1, seq_len - 1, dtype=torch.bool)
    )

    train_collate = make_collate_fn(
        train_dataset,
        src_pad=get_special_tokens(tokenizer_src, "source")[2],
        tgt_pad=tgt_pad,
        causal_mask=causal_mask,
    )
    val_collate = make_collate_fn(
        val_dataset,
        src_pad=get_special_tokens(tokenizer_src, "source")[2],
        tgt_pad=tgt_pad,
        causal_mask=causal_mask,
    )

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=4 if config["num_workers"] > 0 else None,
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=train_collate, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, collate_fn=val_collate, **loader_kwargs
    )

    return (
        train_loader,
        val_loader,
        tokenizer_src,
        tokenizer_tgt
    )


def make_lr_lambda(warmup_steps):
    """Linear warmup then inverse-sqrt decay: transformers need the warmup."""
    warmup_steps = max(warmup_steps, 1)

    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps / step) ** 0.5

    return lr_lambda


def evaluate(model, loader, device, config):
    """Validation loss per valid token (optionally capped at max_val_batches)."""

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    use_amp = config["precision"] == "bf16" and device.type == "cuda"

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            if batch_idx >= config["max_val_batches"]:
                break

            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            decoder_input = batch["decoder_input"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
            decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )
                logits = model.project(decoder_output)

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    label.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=config["label_smoothing"],
                )

            valid_tokens = (label != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    model.train()

    return total_loss / max(total_tokens, 1)


def main():

    config = parse_args()

    torch.manual_seed(config["seed"])

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    (train_loader, val_loader, src_tok, tgt_tok) = create_dataloaders(config)

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    model_config = {
        "src_vocab_sz": src_tok.get_vocab_size(),
        "tgt_vocab_sz": tgt_tok.get_vocab_size(),
        "src_seq_len": config["seq_len"],
        "tgt_seq_len": config["seq_len"],
        "d_model": config["d_model"],
        "N": config["N"],
        "num_head": config["num_head"],
        "dropout": config["dropout"],
        "d_ff": config["d_ff"],
    }

    model = buil_transformer(**model_config).to(device)

    if config["compile"]:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        fused=device.type == "cuda",
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(config["warmup_steps"])
    )

    # bf16 has the same exponent range as fp32, so no loss scaling is needed.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config["precision"] == "fp16"
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_path = Path(config["checkpoint_dir"]) / "best_transformer.pt"

    best_val_loss = float("inf")
    budget_s = config["time_budget_min"] * 60
    start = time.time()
    samples_seen = 0
    out_of_time = False

    print(
        f"\nTime budget: {config['time_budget_min']} min "
        f"| seq_len={config['seq_len']} batch={config['batch_size']} "
        f"d_model={config['d_model']} N={config['N']} "
        f"precision={config['precision']}"
    )

    for epoch in range(config["epochs"]):

        model.train()

        total_train_loss = 0.0
        total_train_tokens = 0
        batches_run = 0
        epoch_start = time.time()

        print(f"\nEpoch [{epoch + 1}/{config['epochs']}]")
        print("-" * 60)

        use_amp = config["precision"] == "bf16" and device.type == "cuda"

        for batch_idx, batch in enumerate(train_loader):

            if time.time() - start > budget_s:
                out_of_time = True
                break

            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            decoder_input = batch["decoder_input"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
            decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )
                logits = model.project(decoder_output)

                device_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    label.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=config["label_smoothing"],
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(device_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            valid_tokens = (label != -100).sum().item()
            total_train_loss += device_loss.item() * valid_tokens
            total_train_tokens += valid_tokens
            batches_run += 1
            samples_seen += encoder_input.size(0)

            if (batch_idx + 1) % config["log_every"] == 0:

                elapsed = time.time() - start
                rate = samples_seen / max(elapsed, 1e-9)

                print(
                    f"step {batch_idx + 1}/{len(train_loader)} "
                    f"| loss {device_loss.item():.4f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e} "
                    f"| {rate:.0f} samp/s "
                    f"| {elapsed / 60:.1f}/{config['time_budget_min']} min"
                )

        train_loss = total_train_loss / max(total_train_tokens, 1)

        if batches_run == 0:
            break

        val_loss = evaluate(model, val_loader, device, config)

        print("\nEpoch Results")
        print("-" * 60)
        print(f"Training Loss:   {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Epoch time:      {(time.time() - epoch_start) / 60:.2f} min")

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "model_config": model_config,
                    "train_config": config,
                },
                best_path,
            )

            print(f"Best model checkpoint saved: {best_path}")

        if out_of_time:
            print("\nTime budget reached.")
            break

    print("\nTraining completed!")
    print(f"Total time:      {(time.time() - start) / 60:.2f} min")
    print(f"Samples seen:    {samples_seen:,}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

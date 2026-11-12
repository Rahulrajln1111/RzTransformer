import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from model import buil_transformer

# --------------------------------------------------
# 1. Configuration (must match training configuration)
# --------------------------------------------------

# train.py writes to checkpoints/, older runs left a copy at the repo root.
CHECKPOINT_CANDIDATES = [
    "checkpoints/best_transformer.pt",
    "best_transformer.pt",
]

SRC_TOKENIZER_PATH = "tokenizer/tokenizer_en.json"
TGT_TOKENIZER_PATH = "tokenizer/tokenizer_hi_ng.json"

parser = argparse.ArgumentParser(description="Translate with a trained checkpoint.")
parser.add_argument(
    "--checkpoint", default=None, help="path to a .pt checkpoint to test"
)
args = parser.parse_args()

if args.checkpoint:
    CHECKPOINT_CANDIDATES = [args.checkpoint]

# Fallback shape, only used for checkpoints saved before model_config existed.
SEQ_LEN = 64
D_MODEL = 512
N = 6
NUM_HEADS = 8
D_FF = 2048
DROPOUT = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# --------------------------------------------------
# 2. Load tokenizers
# --------------------------------------------------

src_tokenizer = Tokenizer.from_file(SRC_TOKENIZER_PATH)
tgt_tokenizer = Tokenizer.from_file(TGT_TOKENIZER_PATH)

src_vocab_size = src_tokenizer.get_vocab_size()
tgt_vocab_size = tgt_tokenizer.get_vocab_size()

print("English vocabulary:", src_vocab_size)
print("Hinglish vocabulary:", tgt_vocab_size)

# --------------------------------------------------
# 3. Load checkpoint
# --------------------------------------------------

CHECKPOINT_PATH = next(
    (p for p in CHECKPOINT_CANDIDATES if Path(p).exists()),
    None,
)

if CHECKPOINT_PATH is None:
    raise FileNotFoundError(
        f"No checkpoint found. Looked in: {CHECKPOINT_CANDIDATES}"
    )

print("Checkpoint:", CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

# Your checkpoint might contain the state dict directly,
# or inside a dictionary under a key.

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
    state_dict = checkpoint["model"]

elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
    state_dict = checkpoint

else:
    raise ValueError(
        f"Cannot identify model weights. Checkpoint keys: {checkpoint.keys()}"
    )

print("\nCheckpoint loaded successfully!")
print("Number of parameter tensors:", len(state_dict))

# --------------------------------------------------
# 4. Reconstruct the Transformer
# --------------------------------------------------

# Newer checkpoints carry their own shape; old ones fall back to the
# constants above. This is why shrinking the model used to break inference.
saved_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None

if saved_config:
    print("Model config from checkpoint:", saved_config)
    SEQ_LEN = saved_config["src_seq_len"]
    model = buil_transformer(**saved_config)
else:
    model = buil_transformer(
        src_vocab_sz=src_vocab_size,
        tgt_vocab_sz=tgt_vocab_size,
        src_seq_len=SEQ_LEN,
        tgt_seq_len=SEQ_LEN,
        d_model=D_MODEL,
        N=N,
        num_head=NUM_HEADS,
        dropout=DROPOUT,
        d_ff=D_FF
    )

# Load trained weights
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("\nTransformer loaded successfully!")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# --------------------------------------------------
# 5. Special token IDs
# --------------------------------------------------

src_sos = src_tokenizer.token_to_id("[SOS]")
src_eos = src_tokenizer.token_to_id("[EOS]")
src_pad = src_tokenizer.token_to_id("[PAD]")

tgt_sos = tgt_tokenizer.token_to_id("[SOS]")
tgt_eos = tgt_tokenizer.token_to_id("[EOS]")
tgt_pad = tgt_tokenizer.token_to_id("[PAD]")

if None in (src_sos, src_eos, src_pad, tgt_sos, tgt_eos, tgt_pad):
    raise ValueError("One or more special tokens are missing from the tokenizers.")

# --------------------------------------------------
# 6. Greedy decoding
# --------------------------------------------------

@torch.inference_mode()
def translate(sentence: str, max_length: int = SEQ_LEN):

    model.eval()

    # Tokenize English sentence
    src_ids = src_tokenizer.encode(sentence).ids

    # Reserve space for SOS and EOS
    src_ids = src_ids[:SEQ_LEN - 2]

    # Add SOS and EOS
    encoder_input = torch.tensor(
        [[src_sos] + src_ids + [src_eos]],
        dtype=torch.long,
        device=device
    )

    # Encoder padding mask: [batch, 1, 1, source_length]
    encoder_mask = (
        (encoder_input != src_pad)
        .unsqueeze(1)
        .unsqueeze(2)
    )

    # Encode source sentence once
    encoder_output = model.encode(
        encoder_input,
        encoder_mask
    )

    # Start decoder with SOS
    generated_ids = [tgt_sos]

    for _ in range(max_length - 1):

        decoder_input = torch.tensor(
            [generated_ids],
            dtype=torch.long,
            device=device
        )

        decoder_length = decoder_input.size(1)

        # Causal mask: decoder cannot see future tokens
        causal_mask = torch.tril(
            torch.ones(
                (1, 1, decoder_length, decoder_length),
                dtype=torch.bool,
                device=device
            )
        )

        # Decoder mask: causal mask AND target padding mask
        target_padding_mask = (
            (decoder_input != tgt_pad)
            .unsqueeze(1)
            .unsqueeze(2)
        )

        decoder_mask = causal_mask & target_padding_mask

        # Decoder
        decoder_output = model.decode(
            encoder_output,
            encoder_mask,
            decoder_input,
            decoder_mask
        )

        # Project hidden states to vocabulary log probabilities
        log_probs = model.project(decoder_output)

        # Select the most probable next token
        next_token = log_probs[0, -1, :].argmax().item()

        # Stop when EOS is generated
        if next_token == tgt_eos:
            break

        generated_ids.append(next_token)

    # Convert generated token IDs into Hinglish text
    translation = tgt_tokenizer.decode(
        generated_ids[1:],
        skip_special_tokens=True
    )

    return translation


# --------------------------------------------------
# 7. Test your trained Transformer
# --------------------------------------------------

if __name__ == "__main__":

    while True:

        try:
            sentence = input("\nEnter English sentence (or 'exit'): ")
        except (EOFError, KeyboardInterrupt):
            # piped input ends here; not an error
            print()
            break

        if sentence.lower().strip() == "exit":
            break

        if not sentence.strip():
            continue

        result = translate(sentence)

        print("\nEnglish :", sentence)
        print("Hinglish:", result)